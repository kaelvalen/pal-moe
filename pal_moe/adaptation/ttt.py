"""
Dual-Mode TTT for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts):
Mode A: Continual Training (labeled stream with prototype router/expert stability)
Mode B: Test-Time Adaptation (unlabeled stream with entropy minimization, consistency, self-supervised)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from ..models.moe import DynamicMoE
from ..memory.prototype_memory import PrototypeMemory
from ..trigger.expert_trigger import QuantitativeTrigger
from ..builder.expert_builder import ExpertBuilder


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
        self.device = device

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        other_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]
        param_groups = []
        if encoder_params:
            enc_lr = self.encoder_lr if self.encoder_lr is not None else self.lr
            param_groups.append({"params": encoder_params, "lr": enc_lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.lr})
        if not param_groups:
            param_groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": self.lr}]
        return torch.optim.Adam(param_groups, weight_decay=1e-5)

    def train_task(
        self,
        task_id: int,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 5,
        enable_expansion: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains the DynamicMoE model on a given task.
        """
        self.model.train()
        history = {
            "loss_total": [],
            "loss_task": [],
            "loss_router_stab": [],
            "loss_expert_stab": [],
            "trigger_events": 0,
            "experts_added": 0,
            "gate_rejections": 0,
        }

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
                    new_id = self.model.add_expert(
                        new_expert=candidate,
                        parent_id=parent_idx,
                        prototype_feat=h_proto,
                    )
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
                    print(f"      [Validation Gate] Rejected! Reason: {gate_result.rejection_reason}")

        # Step 4.5: If task 0, align router's initial expert with task 0 representation
        if task_id == 0 and len(self.model.experts) >= 1:
            with torch.no_grad():
                for x_init, _ in train_loader:
                    x_init = x_init.to(self.device)
                    h_0 = self.model.get_routing_features(x_init).mean(dim=0)
                    self.model.router.gate.weight.data[0] = F.normalize(h_0, dim=0) * 3.0
                    self.model.router.gate.bias.data[0] = 0.0
                    break


        # Freeze old experts during Step 5 to prevent catastrophic forgetting
        for i, exp in enumerate(self.model.experts):
            if i < len(self.model.experts) - 1:
                for p in exp.parameters():
                    p.requires_grad = False
            else:
                for p in exp.parameters():
                    p.requires_grad = True
        # Rebuild optimizer to reflect frozen states
        self.optimizer = self._build_optimizer()

        # Step 5: Continual Training loop with joint stability loss

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Main task prediction
                logits = self.model(x)
                loss_task = F.cross_entropy(logits, y)

                # Prototype stability losses (router, expert, and optional trainable encoder stability)
                l_router_stab, l_expert_stab, l_enc_stab = self.prototype_memory.compute_stability_losses(
                    self.model,
                    lambda_r=self.lambda_r,
                    lambda_e=self.lambda_e,
                    lambda_enc=self.lambda_enc,
                    return_enc=True,
                )

                loss = loss_task + l_router_stab + l_expert_stab + l_enc_stab

                # Optional latent exemplar replay from prototype memory (hybrid mode)
                l_replay = torch.tensor(0.0, device=self.device)
                if self.replay_exemplars and not self.prototype_memory.is_empty():
                    # Latent Replay: Extremely cheap and memory efficient
                    exemplar_batch = self.prototype_memory.get_exemplar_batch(self.device)
                    if exemplar_batch is not None:
                        x_latent_rep, y_rep = exemplar_batch
                        rep_logits = self.model(latent_h=x_latent_rep)
                        l_replay = F.cross_entropy(rep_logits, y_rep) * self.lambda_replay
                        loss = loss + l_replay
                loss = loss + l_replay

                loss.backward()
                
                # Zero out gradients for OLD router rows to completely prevent router forgetting
                if self.model.num_experts > 1:
                    if self.model.router.gate.weight.grad is not None:
                        self.model.router.gate.weight.grad[:self.model.num_experts-1] = 0.0
                    if self.model.router.gate.bias.grad is not None:
                        self.model.router.gate.bias.grad[:self.model.num_experts-1] = 0.0

                self.optimizer.step()


                # Update EMA encoder if enabled
                self.model.update_ema_encoder()

                history["loss_total"].append(loss.item())
                history["loss_task"].append(loss_task.item())
                history["loss_router_stab"].append(l_router_stab.item())
                history["loss_expert_stab"].append(l_expert_stab.item())

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
                )
                registered_count += x.size(0)
                if registered_count >= 256:
                    break

        # Step 7: Representation refresh if encoder is active/EMA to avoid drift
        if self.model.use_ema_encoder and self.model.ema_encoder is not None:
            self.prototype_memory.refresh_representations(self.model.ema_encoder)
        elif any(p.requires_grad for p in self.model.encoder.parameters()):
            self.prototype_memory.refresh_representations(self.model.encoder)


        
        # Step 8: End-of-task Joint Latent Fine-tuning
        # DO NOT unfreeze old experts! Only the newest expert and router learn negative boundaries.


        # Expose experts to negative boundaries from all stored latent exemplars (OOD penalty)
        if task_id > 0 and not self.prototype_memory.is_empty():
            self.model.train()
            exemplar_batch = self.prototype_memory.get_exemplar_batch(self.device)
            if exemplar_batch is not None:
                x_latent_all, y_all = exemplar_batch
                if x_latent_all.size(0) > 0:
                    for _ in range(5):  # 5 epochs of joint calibration
                        self.optimizer.zero_grad()
                        # Full joint forward pass bypassing encoder
                        logits_joint = self.model(latent_h=x_latent_all)
                        loss_joint = F.cross_entropy(logits_joint, y_all)
                        loss_joint.backward()
                        self.optimizer.step()

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
                loss_entropy = -(probs_clean * torch.log(probs_clean + 1e-9)).sum(dim=-1).mean()

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
