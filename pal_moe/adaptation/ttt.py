"""
Dual-Mode TTT for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts):
Mode A: Continual Training (labeled stream with prototype router/expert stability)
Mode B: Test-Time Adaptation (unlabeled stream with entropy minimization, consistency, self-supervised)
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from ..models.moe import DynamicMoE
from ..memory.prototype_memory import PrototypeMemory
from ..trigger.expert_trigger import QuantitativeTrigger
from ..builder.expert_builder import ExpertBuilder
from ..persistence import save_checkpoint


class ContinualTrainer:
    """
    Mode A: Continual Training on labeled task stream.
    Optimizes:
        L = L_new + lambda_r * L_router_stab + lambda_e * L_expert_stab
    Checks quantitative trigger S(x) to build new validated experts.
    """

    def __init__(
        self,
        model: DynamicMoE,
        prototype_memory: PrototypeMemory,
        trigger: QuantitativeTrigger,
        builder: ExpertBuilder,
        lambda_r: float = 1.0,
        lambda_e: float = 1.0,
        lambda_enc: float = 0.0,
        replay_exemplars: bool = False,
        lambda_replay: float = 1.0,
        lr: float = 1e-3,
        encoder_lr: Optional[float] = None,
        max_experts: int = 8,
        joint_unfreeze_all: bool = True,
        joint_keep_routing_lock: bool = False,
        joint_freeze_router: bool = False,
        lambda_ood: float = 0.0,
        router_anchor_steps: int = 0,
        router_anchor_lr: float = 1e-3,
        joint_calib_epochs: int = 5,
        proto_samples: int = 256,
        refresh_anchors_after_calib: bool = False,
        keep_optimizer_state: bool = False,
        checkpoint_dir: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.prototype_memory = prototype_memory
        self.trigger = trigger
        self.builder = builder
        self.lambda_r = lambda_r
        self.lambda_e = lambda_e
        self.lambda_enc = lambda_enc
        self.replay_exemplars = replay_exemplars
        self.lambda_replay = lambda_replay
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.max_experts = max_experts
        # If True, end-of-task joint calibration temporarily unfreezes ALL experts
        # and the full router (historical behavior). If False, only the newest expert
        # and router row learn during calibration (this is what caused catastrophic
        # routing collapse / recency bias on Split-MNIST).
        self.joint_unfreeze_all = joint_unfreeze_all
        # If joint_unfreeze_all is True, keep the historical ROUTING rows locked so
        # old-task inputs keep flowing to their original experts while all experts
        # still calibrate on every stored exemplar (negative boundaries).
        self.joint_keep_routing_lock = joint_keep_routing_lock
        # Strongest protection: during joint calibration the ENTIRE router (including
        # the newest row) is frozen, so routing can never be stolen from historical
        # experts. Only the experts calibrate. Empirically, training the newest row
        # on old exemplars caused a "recency funnel": every task's inputs ended up
        # routed to the newest expert (see routing_asymmetry_debug.json).
        self.joint_freeze_router = joint_freeze_router
        # OOD negative-boundary weight: while a new expert trains on a new task, its
        # predictions on HISTORICAL prototypes are pushed toward maximum entropy
        # (lambda_ood * H). This keeps the newest expert agnostic about old tasks, so
        # old-task inputs have no reason to be routed to it (complements null-space
        # routing anchoring). 0.0 disables the term.
        self.lambda_ood = lambda_ood
        self.router_anchor_steps = router_anchor_steps
        self.router_anchor_lr = router_anchor_lr
        self.joint_calib_epochs = joint_calib_epochs
        self.proto_samples = proto_samples
        self.refresh_anchors_after_calib = refresh_anchors_after_calib
        self.keep_optimizer_state = keep_optimizer_state
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        other_params = [
            p
            for n, p in self.model.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]
        param_groups = []
        if encoder_params:
            enc_lr = self.encoder_lr if self.encoder_lr is not None else self.lr
            param_groups.append({"params": encoder_params, "lr": enc_lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.lr})
        if not param_groups:
            param_groups = [
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": self.lr}
            ]
        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)

        # Optional: carry Adam moments across rebuilds (each task/expansion
        # currently recreates the optimizer, which resets momentum for the
        # router and the historical experts). State is matched by parameter
        # name, so the mapping survives expert/router growth.
        previous = getattr(self, "optimizer", None)
        if self.keep_optimizer_state and previous is not None:
            name_of = {id(p): n for n, p in self.model.named_parameters()}
            old_state = {}
            for group in previous.param_groups:
                for p in group["params"]:
                    if p in previous.state:
                        old_state[name_of.get(id(p))] = previous.state[p]
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = old_state.get(name_of.get(id(p)))
                    if state is not None:
                        optimizer.state[p] = state
        return optimizer

    def _distill_router_anchors(self, steps: int, lr: float) -> float:
        """
        Trains the router to reproduce the explicit prototype owners:
        P(owner expert | prototype feature) — a direct, cross-task supervised
        signal available without any raw exemplars. Historical routing rows are
        temporarily unlocked, then the standard lock is restored.
        """
        owned = [
            p
            for p in self.prototype_memory.prototypes
            if p.owner_expert is not None
        ]
        if not owned or steps <= 0:
            return 0.0

        # Guard: distilling with a trainable encoder would fit stale prototype
        # features (recorded before the encoder moved). Only a frozen encoder (or
        # refreshed prototypes) keeps the targets meaningful.
        if any(p.requires_grad for p in self.model.encoder.parameters()):
            print(
                "      [router-anchor] WARNING: encoder has trainable parameters; "
                "prototype anchors may be stale. Use --freeze_encoder (or refresh "
                "prototypes) for correct distillation."
            )

        vp = torch.stack([p.v_p for p in owned], dim=0).to(self.device)
        y_owner = torch.tensor(
            [int(p.owner_expert) for p in owned], device=self.device
        )
        # Unlock all rows for the distillation, restore the lock afterwards.
        self.model.router.lock_historical_routing(0)
        params = [p for p in self.model.router.parameters() if p.requires_grad]
        if not params:  # non-parametric routers (e.g. distance) cannot be distilled
            self.model.freeze_historical_experts(leave_unfrozen=1)
            self.optimizer = self._build_optimizer()
            return 0.0
        was_training = self.model.router.training
        self.model.router.train()
        optimizer = torch.optim.Adam(params, lr=lr)
        last_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad()
            probs = self.model.router.get_full_distribution(vp)
            loss = F.nll_loss(torch.log(probs + 1e-9), y_owner)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        if not was_training:
            self.model.router.eval()

        self.model.freeze_historical_experts(leave_unfrozen=1)
        self.optimizer = self._build_optimizer()
        return last_loss

    def train_task(
        self,
        task_id: int,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 5,
        enable_expansion: bool = True,
        enable_anchor: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains the DynamicMoE model on a given task.
        """
        self.model.train()
        # A frozen encoder must not drift through BatchNorm running statistics
        # either: keep it in eval mode for the whole task phase.
        if not any(p.requires_grad for p in self.model.encoder.parameters()):
            self.model.encoder.eval()
        history = {
            "loss_total": [],
            "loss_task": [],
            "loss_router_stab": [],
            "loss_expert_stab": [],
            "trigger_events": 0,
            "experts_added": 0,
            "gate_rejections": 0,
            "router_anchor_loss": 0.0,
        }
        # Expert that ends up responsible for this task (explicit prototype anchor)
        task_expert_id: Optional[int] = None

        # Step 1: For tasks > 0, check quantitative trigger early on new task data
        if enable_expansion and task_id > 0:
            trigger_eval = None
            eval_x_list, eval_y_list = [], []
            for x_sample, y_sample in train_loader:
                eval_x_list.append(x_sample)
                eval_y_list.append(y_sample)
                if sum(t.size(0) for t in eval_x_list) >= 256:
                    break

            if eval_x_list:
                x_eval = torch.cat(eval_x_list, dim=0).to(self.device)
                y_eval = torch.cat(eval_y_list, dim=0).to(self.device)
                trigger_eval = self.trigger.evaluate(
                    model=self.model,
                    x=x_eval,
                    y=y_eval,
                    prototype_memory=self.prototype_memory,
                )

            if trigger_eval is not None and trigger_eval.should_trigger:
                history["trigger_events"] += 1
                parent_idx = trigger_eval.best_parent_expert_idx
                parent_expert = self.model.experts[parent_idx]

                # Step 2: Function-preserving expansion
                candidate = self.builder.create_candidate_from_parent(
                    parent_expert=parent_expert,
                    new_expert_id=self.model.num_experts,
                    creation_task=task_id,
                )

                # Step 3: Train candidate on new task with distillation
                train_loss = self.builder.train_candidate(
                    candidate_expert=candidate,
                    encoder=self.model.encoder,
                    train_loader=train_loader,
                    prototype_memory=self.prototype_memory,
                    parent_expert=parent_expert,
                    epochs=3,
                    lr=self.lr,
                    device=self.device,
                )

                # Step 4: Validation Gate
                gate_result = self.builder.validate_candidate(
                    candidate_expert=candidate,
                    parent_expert=parent_expert,
                    encoder=self.model.encoder,
                    val_loader=val_loader,
                    prototype_memory=self.prototype_memory,
                    train_loss=train_loss,
                    device=self.device,
                )

                if gate_result.passed:
                    history["experts_added"] += 1
                    with torch.no_grad():
                        h_proto = self.model.get_routing_features(x_eval).mean(dim=0)
                    # Null-space anchoring basis: historical prototype centroids.
                    # The new routing row is projected onto their null space so it
                    # cannot overlap historical routing regions at initialization.
                    null_basis = (
                        self.prototype_memory.get_prototype_matrix(self.device)
                        if enable_anchor
                        else None
                    )
                    new_id = self.model.add_expert(
                        new_expert=candidate,
                        parent_id=parent_idx,
                        prototype_feat=h_proto,
                        null_space_basis=null_basis,
                    )
                    task_expert_id = new_id
                    # Enforce capacity control if exceeded (with prototype memory synchronization)
                    self.builder.enforce_capacity_control(
                        self.model,
                        prototype_memory=self.prototype_memory,
                        max_experts=self.max_experts,
                    )
                    # Refresh optimizer parameters to include new expert and expanded router
                    self.optimizer = self._build_optimizer()
                else:
                    history["gate_rejections"] += 1
                    # No new expert: the task was trained on the *newest* expert
                    # (freeze_historical_experts leaves only that one trainable),
                    # so that is the expert that actually handles this task.
                    # Anchoring the task's prototypes to it keeps the router
                    # distillation consistent with the model's real allocation.
                    task_expert_id = self.model.num_experts - 1
                    print(
                        f"      [Validation Gate] Rejected! Reason: {gate_result.rejection_reason}"
                    )

        # Step 4.5: If task 0, align router's initial expert with task 0 representation
        if task_id == 0 and len(self.model.experts) >= 1:
            task_expert_id = self.model.num_experts - 1
            with torch.no_grad():
                for x_init, _ in train_loader:
                    x_init = x_init.to(self.device)
                    h_0 = self.model.get_routing_features(x_init).mean(dim=0)
                    if hasattr(self.model.router, "gate"):
                        self.model.router.gate.weight.data[0] = (
                            F.normalize(h_0, dim=0) * 3.0
                        )
                        self.model.router.gate.bias.data[0] = 0.0
                    elif hasattr(self.model.router, "centroids"):
                        self.model.router.centroids[0] = F.normalize(h_0, dim=0)
                    elif hasattr(self.model.router, "keys"):
                        # Cosine-attention router: point the first expert's key at
                        # the task-0 feature direction.
                        self.model.router.keys.data[0] = F.normalize(h_0, dim=0)
                    break

        # Mathematically freeze historical experts and their routing paths
        self.model.freeze_historical_experts(leave_unfrozen=1)
        self.optimizer = self._build_optimizer()

        # Step 5: Continual Training loop with joint stability loss

        # Per-batch losses are accumulated on-device and converted once, instead
        # of four host synchronisations (.item()) per batch.
        _loss_rows: list = []

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Main task prediction
                logits = self.model(x)
                loss_task = F.cross_entropy(logits, y)

                # Prototype stability losses (router, expert, and optional trainable encoder stability)
                l_router_stab, l_expert_stab, l_enc_stab = (
                    self.prototype_memory.compute_stability_losses(
                        self.model,
                        lambda_r=self.lambda_r,
                        lambda_e=self.lambda_e,
                        lambda_enc=self.lambda_enc,
                        return_enc=True,
                    )
                )

                loss = loss_task + l_router_stab + l_expert_stab + l_enc_stab

                # OOD negative-boundary term: the newest expert must stay agnostic
                # about historical tasks, so its predictions on old prototypes are
                # pushed toward maximum entropy. This is what the README's "negative
                # boundaries / OOD penalty" claim refers to: the new expert learns the
                # boundary of ITS OWN task, not a generalist solution.
                l_ood = torch.tensor(0.0, device=self.device)
                if (
                    self.lambda_ood > 0
                    and task_id > 0
                    and not self.prototype_memory.is_empty()
                ):
                    vp_old = self.prototype_memory.get_prototype_matrix(self.device)
                    if vp_old is not None and vp_old.size(0) > 0:
                        with torch.no_grad():
                            vp_old = vp_old.detach()
                        ood_logits = self.model.experts[-1](vp_old, track_usage=False)
                        ood_probs = F.softmax(ood_logits, dim=-1)
                        entropy_ood = (
                            -(ood_probs * torch.log(ood_probs + 1e-9))
                            .sum(dim=-1)
                            .mean()
                        )
                        l_ood = -self.lambda_ood * entropy_ood
                        loss = loss + l_ood

                # Optional latent exemplar replay from prototype memory (hybrid mode)
                l_replay = torch.tensor(0.0, device=self.device)
                if self.replay_exemplars and not self.prototype_memory.is_empty():
                    # Latent Replay: Extremely cheap and memory efficient
                    exemplar_batch = self.prototype_memory.get_exemplar_batch(
                        self.device
                    )
                    if exemplar_batch is not None:
                        x_latent_rep, y_rep = exemplar_batch
                        rep_logits = self.model(latent_h=x_latent_rep)
                        l_replay = (
                            F.cross_entropy(rep_logits, y_rep) * self.lambda_replay
                        )
                        loss = loss + l_replay

                loss.backward()

                self.optimizer.step()

                if hasattr(self.model.router, "update_active_centroid"):
                    h_detached = self.model.get_routing_features(x).detach()
                    self.model.router.update_active_centroid(h_detached)

                # Update EMA encoder if enabled
                self.model.update_ema_encoder()

                _loss_rows.append(
                    torch.stack(
                        [
                            loss.detach(),
                            loss_task.detach(),
                            l_router_stab.detach(),
                            l_expert_stab.detach(),
                        ]
                    )
                )

        if _loss_rows:
            rows = torch.stack(_loss_rows).cpu().tolist()
            history["loss_total"] = [r[0] for r in rows]
            history["loss_task"] = [r[1] for r in rows]
            history["loss_router_stab"] = [r[2] for r in rows]
            history["loss_expert_stab"] = [r[3] for r in rows]

        # Step 6: Register task prototypes into memory with labels and raw exemplars across batches
        self.model.eval()
        with torch.no_grad():
            registered_count = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                h = self.model.get_routing_features(x)
                g_dist = self.model.router.get_full_distribution(h)
                all_expert_outs = self.model.get_all_expert_outputs(h)
                self.prototype_memory.register_task_batch(
                    features=h,
                    routing_dists=g_dist,
                    all_expert_outs=all_expert_outs,
                    task_id=task_id,
                    labels=y,
                    raw_inputs=x,
                    owner_expert=task_expert_id,
                )
                registered_count += x.size(0)
                if registered_count >= self.proto_samples:
                    break

        # Step 7: Representation refresh if encoder is active/EMA to avoid drift
        if self.model.use_ema_encoder and self.model.ema_encoder is not None:
            self.prototype_memory.refresh_representations(self.model.ema_encoder)
        elif any(p.requires_grad for p in self.model.encoder.parameters()):
            self.prototype_memory.refresh_representations(self.model.encoder)

        # Step 8: End-of-task Joint Latent Fine-tuning
        # Joint calibration exposes EVERY stored latent exemplar to the experts so
        # they learn the "negative boundaries" of other tasks (OOD penalty).
        #
        # Routing-protection design (empirically validated, see
        # routing_asymmetry_debug.json):
        #   1. joint_unfreeze_all=False caused recency bias: only the newest row was
        #      calibrated, it captured every task's routing, and the newest expert
        #      (thinly trained on old exemplars) classified them poorly.
        #   2. Even with all rows unlocked, the newest row learned to fire on old
        #      exemplars during calibration and stole historical routing.
        #   => With joint_freeze_router=True the ENTIRE router (including the newest
        #      row) is frozen during calibration: routing is mathematically pinned to
        #      the regions learned during each task-phase, so it can never be stolen.
        #      Only the experts calibrate on the shared exemplar set. The newest row
        #      is re-unlocked afterwards for the next task's training phase.
        if (
            task_id > 0
            and not self.prototype_memory.is_empty()
            and self.joint_calib_epochs > 0
        ):
            self.model.train()
            exemplar_batch = self.prototype_memory.get_exemplar_batch(self.device)
            if exemplar_batch is not None:
                x_latent_all, y_all = exemplar_batch
                if x_latent_all.size(0) > 0:
                    if self.joint_unfreeze_all:
                        if self.joint_freeze_router:
                            # Freeze ALL routing rows (even the newest) during
                            # calibration; only the experts adapt.
                            self.model.unfreeze_all_experts()
                            self.model.router.lock_historical_routing(
                                self.model.num_experts
                            )
                        elif self.joint_keep_routing_lock:
                            # All experts calibrate, but historical routing stays locked:
                            # old-task data keeps flowing to its original expert.
                            self.model.unfreeze_experts_keep_routing_lock()
                        else:
                            self.model.unfreeze_all_experts()
                        self.optimizer = self._build_optimizer()
                    for _ in range(self.joint_calib_epochs):
                        self.optimizer.zero_grad()
                        # Full joint forward pass bypassing encoder
                        logits_joint = self.model(latent_h=x_latent_all)
                        loss_joint = F.cross_entropy(logits_joint, y_all)

                        # OOD negative-boundary term during calibration: the newest
                        # expert must stay agnostic about historical tasks.
                        if self.lambda_ood > 0:
                            vp_old = self.prototype_memory.get_prototype_matrix(
                                self.device
                            )
                            if vp_old is not None and vp_old.size(0) > 0:
                                with torch.no_grad():
                                    vp_old = vp_old.detach()
                                ood_logits = self.model.experts[-1](
                                    vp_old, track_usage=False
                                )
                                ood_probs = F.softmax(ood_logits, dim=-1)
                                entropy_ood = (
                                    -(ood_probs * torch.log(ood_probs + 1e-9))
                                    .sum(dim=-1)
                                    .mean()
                                )
                                loss_joint = loss_joint - self.lambda_ood * entropy_ood

                        loss_joint.backward()
                        self.optimizer.step()

                    if hasattr(self.model.router, "update_active_centroid"):
                        h_detached = self.model.get_routing_features(
                            x_latent_all
                        ).detach()
                        self.model.router.update_active_centroid(h_detached)

                    if self.joint_unfreeze_all:
                        # Re-lock history: only the newest expert and router row stay trainable
                        self.model.freeze_historical_experts(leave_unfrozen=1)
                        self.optimizer = self._build_optimizer()

        # Step 8.5: Refresh prototype anchors with the calibrated model so the next
        # task's stability losses anchor to the model's actual behaviour instead of
        # the (stale) outputs recorded before calibration.
        if self.refresh_anchors_after_calib and not self.prototype_memory.is_empty():
            self.model.eval()
            refreshed = self.prototype_memory.refresh_anchors(self.model)
            print(f"    [anchors] refreshed {refreshed} prototype anchors")

        # Step 9: Distill the router onto the explicit prototype owners.
        # Zero-replay by construction: it only uses stored latent prototypes and
        # the task-derived owner labels. This gives the router the cross-task
        # credit assignment that end-to-end CE cannot learn reliably, and it
        # removes the need for inference-time anchoring.
        if self.router_anchor_steps > 0:
            history["router_anchor_loss"] = self._distill_router_anchors(
                steps=self.router_anchor_steps, lr=self.router_anchor_lr
            )

        # Persist checkpoint after each task if requested
        if self.checkpoint_dir is not None:
            save_checkpoint(
                os.path.join(self.checkpoint_dir, f"task_{task_id}.pt"),
                self.model,
                self.prototype_memory,
                meta={"task_id": task_id, "num_experts": self.model.num_experts},
            )

        return history


class TestTimeAdapter:
    """
    Mode B: Unlabeled Test-Time Adaptation (TTT).
    During inference on an unlabeled test stream:
    - Shared encoder is FROZEN during adaptation.
    - Only the router and selected expert are updated.
    - Loss = Entropy Minimization + Prediction Consistency + Self-Supervised.
    - Preserves encoder gradients and model state upon completion.
    """

    __test__ = False

    def __init__(
        self,
        model: DynamicMoE,
        steps: int = 1,
        lr: float = 1e-4,
        consistency_weight: float = 0.5,
        noise_std: float = 0.05,
    ):
        self.model = model
        self.steps = steps
        self.lr = lr
        self.consistency_weight = consistency_weight
        self.noise_std = noise_std

    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs test-time adaptation on unlabeled batch x and returns predictions.
        Guarantees that encoder requires_grad and training mode are restored.
        """
        orig_encoder_grads = [p.requires_grad for p in self.model.encoder.parameters()]
        orig_mode = self.model.training

        try:
            # Ensure encoder is frozen during test-time adaptation
            self.model.encoder.eval()
            for p in self.model.encoder.parameters():
                p.requires_grad = False

            # Make router and experts adaptable
            trainable_params = list(self.model.router.parameters())
            for exp in self.model.experts:
                trainable_params.extend(list(exp.parameters()))

            optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

            for _ in range(self.steps):
                optimizer.zero_grad()

                # 1. Forward original
                logits_clean = self.model(x)
                probs_clean = F.softmax(logits_clean, dim=-1)

                # 2. Entropy minimization loss: pushes confident predictions
                loss_entropy = (
                    -(probs_clean * torch.log(probs_clean + 1e-9)).sum(dim=-1).mean()
                )

                # 3. Consistency loss under slight input perturbation
                x_perturbed = x + torch.randn_like(x) * self.noise_std
                logits_noisy = self.model(x_perturbed)
                probs_noisy = F.softmax(logits_noisy, dim=-1)
                loss_consistency = F.mse_loss(probs_clean, probs_noisy)

                loss_ttt = loss_entropy + self.consistency_weight * loss_consistency
                loss_ttt.backward()
                optimizer.step()

            # Final prediction after adaptation
            self.model.eval()
            with torch.no_grad():
                final_logits = self.model(x)
            return final_logits

        finally:
            # Restore original encoder requires_grad flags and training mode
            for p, req in zip(self.model.encoder.parameters(), orig_encoder_grads):
                p.requires_grad = req
            self.model.train(orig_mode)
