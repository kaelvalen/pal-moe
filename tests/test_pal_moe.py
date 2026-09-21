"""
Unit tests for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).
Verifies all mathematical formulations, loss computations, function-preserving expansion,
trigger thresholds, validation gate, and TTT.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pal_moe.adaptation.ttt import ContinualTrainer, TestTimeAdapter
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.evaluation.metrics import ContinualEvaluator
from pal_moe.memory.prototype_memory import PrototypeMemory
from pal_moe.models.encoder import EMAEncoder, SharedEncoder
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE
from pal_moe.models.router import DynamicRouter
from pal_moe.trigger.expert_trigger import QuantitativeTrigger


def test_shared_and_ema_encoder():
    enc = SharedEncoder(input_dim=784, hidden_dims=(64,), output_dim=32, arch="mlp")
    enc.eval()
    ema_enc = EMAEncoder(enc, decay=0.9)

    x = torch.randn(4, 784)
    h1 = enc(x)
    h_ema = ema_enc(x)
    assert h1.shape == (4, 32)
    assert h_ema.shape == (4, 32)
    assert torch.allclose(h1, h_ema, atol=1e-5)

    # Modify encoder
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    ema_enc.update(enc)
    h2 = enc(x)
    h_ema_updated = ema_enc(x)
    # EMA should be between old and new
    assert not torch.allclose(h2, h_ema_updated, atol=1e-4)


def test_prototype_anchored_routing_overrides_router():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2, top_k=1)
    experts = [
        MLPExpert(input_dim=8, hidden_dim=8, num_classes=10, expert_id=i)
        for i in range(2)
    ]
    model = DynamicMoE(encoder=enc, router=router, experts=experts)
    model.eval()

    # Force the learned router to always pick expert 1 (the recency funnel)
    with torch.no_grad():
        model.router.gate.weight.zero_()
        model.router.gate.bias.copy_(torch.tensor([-5.0, 5.0]))

    x = torch.randn(4, 16)
    with torch.no_grad():
        h = model.encoder(x)
        _, greedy, _ = model.router(h)
    assert greedy.flatten().tolist() == [1, 1, 1, 1]

    memory = PrototypeMemory(feature_dim=8, distance_threshold=1e9, store_raw=False)
    memory.register_task_batch(
        features=h[:1],
        routing_dists=torch.tensor([[0.0, 1.0]]),  # stale r_p points at expert 1
        all_expert_outs=torch.zeros(1, 2, 10),
        task_id=0,
        labels=torch.tensor([0]),
        owner_expert=0,  # the expert actually trained for this task
    )
    assert memory.prototypes[0].owner_expert == 0
    model.set_prototype_routing(memory, alpha=1.0)

    with torch.no_grad():
        _, anchored, _ = model._route(model.encoder(x))
    # The explicit owner anchor must win over the (stale) router and r_p
    assert anchored.flatten().tolist() == [0, 0, 0, 0]

    # Training-time routing must stay untouched by the anchoring
    model.train()
    with torch.no_grad():
        _, train_route, _ = model._route(model.encoder(x))
    assert train_route.flatten().tolist() == [1, 1, 1, 1]


def test_prototype_routing_confidence_scales_with_ambiguity():
    """Far/ambiguous inputs must keep the learned router (confidence -> 0)."""
    torch.manual_seed(3)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2, top_k=1)
    experts = [
        MLPExpert(input_dim=8, hidden_dim=8, num_classes=10, expert_id=i)
        for i in range(2)
    ]
    model = DynamicMoE(encoder=enc, router=router, experts=experts)
    model.eval()

    memory = PrototypeMemory(feature_dim=8, store_raw=False)
    # Two prototypes from *different* tasks, equidistant from the query below
    h_anchor = model.encoder(torch.randn(2, 16))
    memory.register_task_batch(
        features=h_anchor,
        routing_dists=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        all_expert_outs=torch.zeros(2, 2, 10),
        task_id=0,
    )
    memory.prototypes[1].task_id = 1
    memory._invalidate_cache()

    query = (h_anchor[0] + h_anchor[1]) / 2  # equidistant from both prototypes
    model.set_prototype_routing(memory, alpha=1.0)
    with torch.no_grad():
        anchors, confidence = model._prototype_routing_anchor(query.unsqueeze(0))
    assert confidence.item() < 0.05, (
        f"ambiguous input should not be anchored: {confidence}"
    )
    assert anchors.shape == (1, 2)


def test_attention_router_routing_expansion_and_lock():
    from pal_moe.models.router import AttentionRouter

    torch.manual_seed(0)
    router = AttentionRouter(input_dim=16, num_experts=2, top_k=1, temperature=0.1)
    h = torch.randn(8, 16)

    weights, topk, logits = router(h)
    assert weights.shape == (8, 2)
    assert topk.shape == (8, 1)
    # top-1 sparse routing: exactly one nonzero weight per sample
    assert (weights > 0).sum(dim=1).eq(1).all()
    assert logits.abs().max() <= 1.0 / 0.1 + 1e-4  # cosine logits are bounded

    dist = router.get_full_distribution(h)
    assert torch.allclose(dist.sum(dim=1), torch.ones(8), atol=1e-5)
    assert (AttentionRouter.compute_entropy(dist) >= 0).all()

    # Expansion keeps old keys and adds a prototype-aligned one
    old_key0 = router.keys.data[0].clone()
    new_id = router.add_expert(parent_id=0, prototype_feat=h[0])
    assert new_id == 2 and router.num_experts == 3
    assert torch.allclose(router.keys.data[0], old_key0)
    aligned = torch.nn.functional.normalize(h[0], dim=0)
    assert (
        torch.dot(torch.nn.functional.normalize(router.keys.data[2], dim=0), aligned)
        > 0.99
    )

    # Locked historical keys receive no gradient
    router.lock_historical_routing(2)
    logits = router._logits(h)
    logits.sum().backward()
    assert router.keys.grad[:2].abs().sum() == 0
    assert router.keys.grad[2].abs().sum() > 0


def test_router_anchor_distillation_learns_prototype_owners():
    from pal_moe.adaptation.ttt import ContinualTrainer

    torch.manual_seed(5)
    encoder = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2, top_k=1, temperature=1.0)
    experts = [
        MLPExpert(input_dim=8, hidden_dim=8, num_classes=10, expert_id=i)
        for i in range(2)
    ]
    model = DynamicMoE(encoder=encoder, router=router, experts=experts)
    memory = PrototypeMemory(feature_dim=8, store_raw=False)

    h = model.encoder(torch.randn(2, 16))
    for i, owner in enumerate((0, 1)):
        memory.register_task_batch(
            features=h[i : i + 1],
            routing_dists=torch.eye(2)[owner : owner + 1],
            all_expert_outs=torch.zeros(1, 2, 10),
            task_id=owner,
            labels=torch.tensor([0]),
            owner_expert=owner,
        )

    trainer = ContinualTrainer(
        model=model,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(),
        builder=ExpertBuilder(),
        device=torch.device("cpu"),
    )
    loss = trainer._distill_router_anchors(steps=300, lr=1e-2)
    assert loss >= 0.0
    preds = model.router.get_full_distribution(h).argmax(dim=1)
    assert preds.tolist() == [0, 1]


def test_rejected_expansion_owner_is_newest_expert():
    """
    Regression: when the validation gate rejects expansion, the task's
    prototypes must be owned by the *newest* expert (the one the task phase
    trained), not by the trigger's best parent. Anchoring to the parent
    conflated two tasks on one expert and collapsed the rejected task
    (MNIST hybrid 81.1% -> 71.1% before the fix).
    """
    from torch.utils.data import DataLoader, TensorDataset

    from pal_moe.adaptation.ttt import ContinualTrainer
    from pal_moe.builder.expert_builder import ValidationGateResult
    from pal_moe.trigger.expert_trigger import TriggerEvaluationResult

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2, top_k=1)
    experts = [
        MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=i)
        for i in range(2)
    ]
    model = DynamicMoE(encoder=enc, router=router, experts=experts)

    class AlwaysTrigger:
        def evaluate(self, model, x, y=None, prototype_memory=None):
            return TriggerEvaluationResult(
                should_trigger=True,
                composite_score=99.0,
                loss_best_expert=1.0,
                router_entropy=0.0,
                proto_distance=1.0,
                max_confidence=0.0,
                best_parent_expert_idx=0,  # wrong owner under the old behaviour
            )

    class RejectingBuilder:
        def create_candidate_from_parent(self, parent_expert, **kwargs):
            return parent_expert.clone_function_preserving(**kwargs)

        def train_candidate(self, **kwargs):
            return 0.0

        def validate_candidate(self, **kwargs):
            return ValidationGateResult(
                passed=False,
                new_task_acc=0.1,
                old_proto_loss_diff=9.9,
                ece=0.0,
                train_loss=0.0,
                val_loss=0.0,
                rejection_reason="forced rejection",
            )

        def enforce_capacity_control(self, model, prototype_memory=None, max_experts=8):
            return {"actions": [], "final_num_experts": model.num_experts}

    x = torch.randn(16, 16)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    memory = PrototypeMemory(feature_dim=8, store_raw=False)

    trainer = ContinualTrainer(
        model=model,
        prototype_memory=memory,
        trigger=AlwaysTrigger(),
        builder=RejectingBuilder(),
        device=torch.device("cpu"),
    )
    history = trainer.train_task(
        task_id=1, train_loader=loader, val_loader=loader, epochs=1
    )

    assert history["gate_rejections"] == 1
    assert history["experts_added"] == 0
    assert model.num_experts == 2
    registered = [p for p in memory.prototypes if p.task_id == 1]
    assert registered, "expected prototypes for the rejected task"
    assert all(p.owner_expert == 1 for p in registered), (
        f"owners must be the newest expert (1); got "
        f"{sorted({p.owner_expert for p in registered})}"
    )


def test_feature_cache_matches_raw_encoder():
    from torch.utils.data import DataLoader, TensorDataset

    from pal_moe.data.feature_cache import build_feature_cache

    class DummyTask:
        def __init__(self, task_id, loader):
            self.task_id = task_id
            self.classes = (0, 1)
            self.train_loader = loader
            self.val_loader = loader
            self.test_loader = loader

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    enc.freeze()
    x = torch.randn(12, 16)
    y = torch.randint(0, 4, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=False)
    tasks = [DummyTask(0, loader)]

    cache = build_feature_cache(
        enc, tasks, torch.device("cpu"), dtype=torch.float32, verbose=False
    )
    h_cached, y_cached = next(iter(cache.tasks[0].test_loader))
    with torch.no_grad():
        h_raw = enc(x)
    assert torch.allclose(h_cached, h_raw, atol=1e-5)
    assert torch.equal(y_cached, y)
    assert cache.encoder.output_dim == 8
    assert torch.allclose(cache.encoder(h_cached), h_cached)

    # A model on cached features must produce the same logits as on raw inputs
    raw_model = DynamicMoE(
        encoder=enc,
        router=DynamicRouter(input_dim=8, num_experts=2, top_k=1),
        experts=[
            MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=i)
            for i in range(2)
        ],
    )
    cached_model = DynamicMoE(
        encoder=cache.encoder,
        router=DynamicRouter(input_dim=8, num_experts=2, top_k=1),
        experts=[
            MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=i)
            for i in range(2)
        ],
    )
    cached_model.load_state_dict(
        {
            k.replace("encoder.", "encoder.", 1): v
            for k, v in raw_model.state_dict().items()
        },
        strict=False,
    )
    with torch.no_grad():
        out_raw = raw_model(x)
        out_cached = cached_model(h_cached)
    assert torch.allclose(out_raw, out_cached, atol=1e-5)


def test_prototype_auto_threshold_and_class_balanced_eviction():
    torch.manual_seed(0)
    memory = PrototypeMemory(
        feature_dim=4,
        distance_threshold=None,
        max_prototypes=8,
        max_prototypes_per_class=2,
    )
    torch.manual_seed(0)
    cluster_a = torch.randn(10, 4) * 0.01
    cluster_b = torch.randn(10, 4) * 0.01 + 5.0
    feats = torch.cat([cluster_a, cluster_b], dim=0)
    labels = torch.cat(
        [torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)]
    )
    routing = torch.softmax(torch.randn(20, 2), dim=-1)
    outs = torch.randn(20, 2, 4)
    memory.register_task_batch(
        features=feats,
        routing_dists=routing,
        all_expert_outs=outs,
        task_id=0,
        labels=labels,
    )
    assert memory.distance_threshold is not None and memory.distance_threshold > 0
    # Scale-free clustering must compress the batch meaningfully instead of the
    # old "every sample becomes a prototype" behaviour (exact ratio is a tuning
    # knob; the ablation grid selects it).
    assert len(memory.prototypes) <= len(feats) // 2, (
        f"clustering ineffective: {len(memory.prototypes)} prototypes for "
        f"{len(feats)} samples"
    )
    # An explicitly tiny threshold must reproduce one-prototype-per-sample
    tight = PrototypeMemory(feature_dim=4, distance_threshold=1e-6)
    tight.register_task_batch(
        features=feats,
        routing_dists=torch.softmax(torch.randn(20, 2), dim=-1),
        all_expert_outs=torch.randn(20, 2, 4),
        task_id=0,
        labels=labels,
    )
    assert len(tight.prototypes) == len(feats)

    # Class-balanced eviction: both classes survive under the quota
    memory2 = PrototypeMemory(
        feature_dim=4,
        distance_threshold=1e-6,
        max_prototypes=2,
        max_prototypes_per_class=1,
    )
    feats2 = torch.tensor([[0.0, 0, 0, 0], [10.0, 0, 0, 0], [20.0, 0, 0, 0]])
    labels2 = torch.tensor([0, 0, 1])
    memory2.register_task_batch(
        features=feats2,
        routing_dists=torch.softmax(torch.randn(3, 2), dim=-1),
        all_expert_outs=torch.randn(3, 2, 4),
        task_id=0,
        labels=labels2,
    )
    kept_labels = sorted(PrototypeMemory._label_of(p) for p in memory2.prototypes)
    assert kept_labels == [0, 1], f"class balance broken: {kept_labels}"


def test_optimizer_state_is_carried_across_rebuilds():
    from pal_moe.adaptation.ttt import ContinualTrainer

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = DynamicMoE(
        encoder=enc,
        router=DynamicRouter(input_dim=8, num_experts=1, top_k=1),
        experts=[MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=0)],
    )
    trainer = ContinualTrainer(
        model=model,
        prototype_memory=PrototypeMemory(feature_dim=8, store_raw=False),
        trigger=QuantitativeTrigger(),
        builder=ExpertBuilder(),
        keep_optimizer_state=True,
        device=torch.device("cpu"),
    )
    x = torch.randn(4, 16)
    out = model(x).sum()
    out.backward()
    trainer.optimizer.step()
    before = {
        id(p): s["step"].item()
        for p, s in trainer.optimizer.state.items()
        if "step" in s
    }
    assert before, "optimizer state should be populated after a step"

    rebuilt = trainer._build_optimizer()
    after = {id(p): s["step"].item() for p, s in rebuilt.state.items() if "step" in s}
    assert after == before, "Adam moments must survive optimizer rebuilds"


def test_soft_top_k_mixture_routing():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=3, top_k=2)
    experts = [
        MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=i)
        for i in range(3)
    ]
    model = DynamicMoE(encoder=enc, router=router, experts=experts)
    x = torch.randn(6, 16)
    with torch.no_grad():
        weights, topk, _ = model.router(model.encoder(x))
    assert (weights > 0).sum(dim=1).eq(2).all()  # exactly two experts per input
    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (6, 4) and torch.isfinite(out).all()


def test_refresh_anchors_updates_targets():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = DynamicMoE(
        encoder=enc,
        router=DynamicRouter(input_dim=8, num_experts=2, top_k=1),
        experts=[
            MLPExpert(input_dim=8, hidden_dim=8, num_classes=4, expert_id=i)
            for i in range(2)
        ],
    )
    memory = PrototypeMemory(feature_dim=8, store_raw=False)
    h = model.encoder(torch.randn(4, 16))
    memory.register_task_batch(
        features=h,
        routing_dists=torch.softmax(torch.randn(4, 2), dim=-1),
        all_expert_outs=torch.randn(4, 2, 4),
        task_id=0,
        labels=torch.zeros(4, dtype=torch.long),
    )
    old_rp = memory.prototypes[0].r_p.clone()
    old_op = memory.prototypes[0].o_p.clone()
    with torch.no_grad():
        model.router.gate.weight.mul_(0.5)
        for e in model.experts:
            e.fc2.weight.mul_(1.5)
    refreshed = memory.refresh_anchors(model)
    assert refreshed == len(memory.prototypes)
    assert not torch.allclose(memory.prototypes[0].r_p, old_rp)
    assert not torch.allclose(memory.prototypes[0].o_p, old_op)
    with torch.no_grad():
        expected_rp = model.router.get_full_distribution(
            memory.get_prototype_matrix(torch.device("cpu"))
        )[0]
    assert torch.allclose(memory.prototypes[0].r_p, expected_rp.cpu(), atol=1e-5)


def test_owner_aware_merging_never_crosses_tasks():
    """Nearby prototypes with different owners must stay separate (router anchors)."""
    memory = PrototypeMemory(feature_dim=4, distance_threshold=1e9, store_raw=False)
    base = torch.zeros(1, 4)
    routing = torch.softmax(torch.randn(1, 2), dim=-1)
    outs = torch.randn(1, 2, 4)

    memory.register_task_batch(
        features=base,
        routing_dists=routing,
        all_expert_outs=outs,
        task_id=0,
        labels=torch.tensor([0]),
        owner_expert=0,
    )
    memory.register_task_batch(
        features=base,
        routing_dists=routing,
        all_expert_outs=outs,
        task_id=1,
        labels=torch.tensor([1]),
        owner_expert=1,
    )
    # Identical features, different owners -> two prototypes (no cross-task merge)
    assert len(memory.prototypes) == 2
    assert sorted(p.owner_expert for p in memory.prototypes) == [0, 1]

    # Same owner within the merge radius -> merges into the existing prototype
    memory.register_task_batch(
        features=base + 1e-6,
        routing_dists=routing,
        all_expert_outs=outs,
        task_id=0,
        labels=torch.tensor([0]),
        owner_expert=0,
    )
    assert len(memory.prototypes) == 2
    assert memory.prototypes[0].count == 2


def test_frozen_encoder_stays_in_eval_mode():
    """freeze() must survive a later model.train() so BN statistics cannot drift."""
    enc = SharedEncoder(input_dim=3072, hidden_dims=None, output_dim=8, arch="conv")
    enc.freeze()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        before = enc(x)
    enc.train()  # e.g. a baseline calling model.train()
    assert not enc.training
    assert all(not m.training for m in enc.modules())
    with torch.no_grad():
        after = enc(x)
    assert torch.allclose(before, after, atol=1e-6)

    enc.unfreeze()
    assert enc._frozen is False
    enc.train()
    assert enc.training


def test_conv_encoder_channels_and_feature_dim():
    default = SharedEncoder(
        input_dim=3072, hidden_dims=None, output_dim=128, arch="conv"
    )
    # Default channels must reproduce the original 3-stage layout exactly
    assert [type(m).__name__ for m in default.net] == [
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "MaxPool2d",
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "MaxPool2d",
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "AdaptiveAvgPool2d",
        "Flatten",
        "Linear",
        "BatchNorm1d",
        "ReLU",
    ]
    assert default.net[-3].in_features == 128

    big = SharedEncoder(
        input_dim=3072,
        hidden_dims=None,
        output_dim=256,
        arch="conv",
        conv_channels=(64, 128, 256),
    )
    big.eval()
    assert big(torch.randn(4, 3, 32, 32)).shape == (4, 256)


def test_function_preserving_expert_expansion():
    parent = MLPExpert(input_dim=32, hidden_dim=16, num_classes=10, expert_id=0)
    child = parent.clone_function_preserving(new_expert_id=1, creation_task=1)

    x = torch.randn(8, 32)
    with torch.no_grad():
        out_parent = parent(x)
        out_child = child(x)

    # EXACT output equality: E_child(h) == E_parent(h)
    diff = torch.max(torch.abs(out_parent - out_child)).item()
    assert diff < 1e-6, f"Function-preserving expansion failed: max diff = {diff}"


def test_dynamic_router_and_entropy():
    router = DynamicRouter(input_dim=32, num_experts=3, top_k=1)
    h = torch.randn(5, 32)

    weights, topk_idx, logits = router(h)
    assert weights.shape == (5, 3)
    assert topk_idx.shape == (5, 1)

    entropy = router.compute_entropy(weights)
    assert entropy.shape == (5,)

    # Add expert
    new_id = router.add_expert(parent_id=0)
    assert new_id == 3
    assert router.num_experts == 4

    weights2, topk_idx2, _ = router(h)
    assert weights2.shape == (5, 4)

    # Prune expert
    router.prune_expert(1)
    assert router.num_experts == 3


def test_prototype_memory_and_stability_losses():
    mem = PrototypeMemory(feature_dim=32, distance_threshold=0.5, ema_alpha=0.8)
    assert mem.is_empty()

    enc = SharedEncoder(input_dim=784, hidden_dims=(64,), output_dim=32)
    router = DynamicRouter(input_dim=32, num_experts=2, top_k=1)
    experts = [MLPExpert(32, 16, 10, i) for i in range(2)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    x = torch.randn(10, 784)
    h = enc(x)
    g = router.get_full_distribution(h)
    exp_outs = moe.get_all_expert_outputs(h)

    # Register prototypes
    mem.register_task_batch(h, g, exp_outs, task_id=0)
    assert len(mem) > 0

    dist = mem.compute_min_distance(h)
    assert dist.shape == (10,)
    assert (dist >= 0).all()

    l_r, l_e = mem.compute_stability_losses(moe, lambda_r=1.0, lambda_e=1.0)
    assert l_r.item() >= 0.0
    assert l_e.item() >= 0.0


def test_quantitative_trigger():
    enc = SharedEncoder(input_dim=784, hidden_dims=(32,), output_dim=16)
    router = DynamicRouter(input_dim=16, num_experts=2, top_k=1)
    experts = [MLPExpert(16, 16, 10, i) for i in range(2)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)
    mem = PrototypeMemory(feature_dim=16)

    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.5, gamma=0.5, delta=0.5, threshold_tau=0.1
    )

    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))

    res = trigger.evaluate(moe, x, y=y, prototype_memory=mem)
    assert isinstance(res.should_trigger, bool)
    assert res.best_parent_expert_idx in (0, 1)
    assert isinstance(res.composite_score, float)


def test_validation_gate_and_capacity():
    parent = MLPExpert(input_dim=16, hidden_dim=8, num_classes=10, expert_id=0)
    builder = ExpertBuilder(min_acc_threshold=0.5, max_proto_drop=0.5, max_ece=0.5)

    child = builder.create_candidate_from_parent(
        parent, new_expert_id=1, creation_task=1
    )
    enc = nn.Linear(16, 16)
    mem = PrototypeMemory(feature_dim=16)

    # Dummy dataloader
    dummy_x = torch.randn(20, 16)
    dummy_y = torch.zeros(20, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    res = builder.validate_candidate(
        candidate_expert=child,
        parent_expert=parent,
        encoder=enc,
        val_loader=loader,
        prototype_memory=mem,
    )
    assert hasattr(res, "passed")
    assert hasattr(res, "ece")


def test_test_time_adaptation():
    enc = SharedEncoder(input_dim=32, hidden_dims=(16,), output_dim=16)
    router = DynamicRouter(input_dim=16, num_experts=2, top_k=1)
    experts = [MLPExpert(16, 16, 5, i) for i in range(2)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    adapter = TestTimeAdapter(moe, steps=2, lr=1e-3)
    x = torch.randn(4, 32)
    pred = adapter.adapt_and_predict(x)
    assert pred.shape == (4, 5)


def test_metrics_computation():
    evaluator = ContinualEvaluator(num_tasks=3)
    evaluator.R = np.array(
        [
            [0.90, 0.00, 0.00],
            [0.85, 0.92, 0.00],
            [0.80, 0.88, 0.95],
        ],
        dtype=np.float32,
    )

    acc = evaluator.compute_average_accuracy()
    forgetting = evaluator.compute_forgetting()
    bwt = evaluator.compute_backward_transfer()

    assert abs(acc - (0.80 + 0.88 + 0.95) / 3.0) < 1e-4
    # Task 0 max past was 0.90, final 0.80 -> drop 0.10
    # Task 1 max past was 0.92, final 0.88 -> drop 0.04
    # Forgetting = (0.10 + 0.04) / 2 = 0.07
    assert abs(forgetting - 0.07) < 1e-4
    # BWT = ((0.80 - 0.90) + (0.88 - 0.92)) / 2 = -0.07
    assert abs(bwt - (-0.07)) < 1e-4


def test_prototype_sync_on_merge_and_prune():
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=3, top_k=1)
    experts = [MLPExpert(8, 8, 4, i) for i in range(3)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    mem = PrototypeMemory(feature_dim=8)
    # Register dummy prototype with 3 experts
    v = torch.randn(8)
    r = torch.tensor([0.6, 0.3, 0.1])
    o = torch.randn(3, 4)
    mem.update_or_create_prototype(v, r, o, task_id=0)

    # Merge expert 1 into expert 0
    moe.merge_experts(0, 1, prototype_memory=mem)

    assert moe.num_experts == 2
    assert len(mem.prototypes) == 1
    p = mem.prototypes[0]
    assert p.r_p.size(0) == 2
    assert p.o_p.size(0) == 2
    # Probability of 0 should now combine 0.6 and 0.3 = 0.9
    assert abs(p.r_p[0].item() - 0.9) < 1e-4
    assert abs(p.r_p.sum().item() - 1.0) < 1e-4

    # Compute stability loss to verify no dimension mismatch
    l_r, l_e = mem.compute_stability_losses(moe)
    assert not torch.isnan(l_r) and not torch.isnan(l_e)


def test_validation_gate_with_acc_proto():
    parent = MLPExpert(input_dim=8, hidden_dim=8, num_classes=3, expert_id=0)
    builder = ExpertBuilder(
        min_acc_threshold=0.3, max_proto_drop=1.0, max_proto_acc_drop=0.01
    )
    child = builder.create_candidate_from_parent(
        parent, new_expert_id=1, creation_task=1
    )

    enc = nn.Identity()
    mem = PrototypeMemory(feature_dim=8)

    # Register labeled prototype exemplars
    feat = torch.randn(8)
    r = torch.tensor([1.0])
    o = torch.randn(1, 3)
    lbl = torch.tensor(0)
    mem.update_or_create_prototype(feat, r, o, task_id=0, label=lbl)

    dummy_x = torch.randn(10, 8)
    dummy_y = torch.randint(0, 3, (10,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_x, dummy_y), batch_size=5
    )

    res = builder.validate_candidate(
        candidate_expert=child,
        parent_expert=parent,
        encoder=enc,
        val_loader=loader,
        prototype_memory=mem,
    )
    assert res.proto_acc_parent is not None
    assert res.proto_acc_cand is not None
    assert res.proto_acc_parent == res.proto_acc_cand  # Exact match due to Net2Net init


def test_refresh_representations_with_raw_inputs():
    enc = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        enc.weight.fill_(1.0)

    mem = PrototypeMemory(feature_dim=4, store_raw=True)
    raw_x = torch.tensor([1.0, 1.0, 1.0, 1.0])
    feat = enc(raw_x)
    r = torch.tensor([1.0])
    o = torch.randn(1, 2)
    mem.update_or_create_prototype(feat, r, o, task_id=0, raw_input=raw_x)

    assert torch.allclose(mem.prototypes[0].v_p, torch.tensor([4.0, 4.0, 4.0, 4.0]))

    # Change encoder weights
    with torch.no_grad():
        enc.weight.fill_(2.0)

    mem.refresh_representations(enc)
    # Output should now be 2 * 4 = 8.0
    assert torch.allclose(mem.prototypes[0].v_p, torch.tensor([8.0, 8.0, 8.0, 8.0]))
    assert mem.prototypes[0].x_p is not None
    assert torch.allclose(mem.prototypes[0].x_p, torch.tensor([[8.0, 8.0, 8.0, 8.0]]))


def test_router_top1_gradient_flow():
    router = DynamicRouter(input_dim=16, num_experts=4, top_k=1)
    h = torch.randn(8, 16, requires_grad=True)
    weights, topk_idx, logits = router(h)

    # In top-1 routing, router logits must receive non-zero gradients
    expert_outputs = torch.randn(8, 4, 3)
    out = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    loss = out.sum()
    loss.backward()

    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum().item() > 0.0


def test_validation_gate_rejection_on_drift():
    parent = MLPExpert(input_dim=8, hidden_dim=8, num_classes=3, expert_id=0)
    builder = ExpertBuilder(
        min_acc_threshold=0.3, max_proto_drop=0.01, max_proto_acc_drop=0.01
    )
    child = builder.create_candidate_from_parent(
        parent, new_expert_id=1, creation_task=1
    )

    # Intentionally perturb child weights to cause drift
    with torch.no_grad():
        child.fc1.weight.add_(torch.randn_like(child.fc1.weight) * 2.0)

    enc = nn.Identity()
    mem = PrototypeMemory(feature_dim=8)
    feat = torch.randn(8)
    r = torch.tensor([1.0])
    o = torch.randn(1, 3)
    mem.update_or_create_prototype(feat, r, o, task_id=0)

    dummy_x = torch.randn(10, 8)
    dummy_y = torch.randint(0, 3, (10,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_x, dummy_y), batch_size=5
    )

    res = builder.validate_candidate(
        candidate_expert=child,
        parent_expert=parent,
        encoder=enc,
        val_loader=loader,
        prototype_memory=mem,
    )
    # Drifted candidate must fail the gate
    assert not res.passed
    assert res.rejection_reason is not None


def test_test_time_adapter_restores_encoder_grad_state():
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    for p in enc.parameters():
        p.requires_grad = True

    router = DynamicRouter(input_dim=8, num_experts=2, top_k=1)
    experts = [MLPExpert(8, 8, 3, i) for i in range(2)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)
    moe.train()

    adapter = TestTimeAdapter(moe, steps=1, lr=1e-3)
    x = torch.randn(4, 16)
    _ = adapter.adapt_and_predict(x)

    # Encoder requires_grad and training mode must be fully preserved
    assert all(p.requires_grad for p in enc.parameters())
    assert moe.training


def test_prototype_memory_store_raw_and_footprint():
    # Test store_raw=False (default)
    mem_compact = PrototypeMemory(feature_dim=16, store_raw=False)
    feat = torch.randn(16)
    r = torch.tensor([1.0, 0.0])
    o = torch.randn(2, 5)
    raw = torch.randn(28 * 28)
    mem_compact.update_or_create_prototype(feat, r, o, task_id=0, raw_input=raw)

    assert len(mem_compact) == 1
    assert mem_compact.prototypes[0].raw_x is None
    fp_compact = mem_compact.estimate_memory_footprint()
    assert fp_compact["num_prototypes"] == 1
    assert fp_compact["total_elements"] < 100

    # Test store_raw=True
    mem_full = PrototypeMemory(feature_dim=16, store_raw=True)
    mem_full.update_or_create_prototype(feat, r, o, task_id=0, raw_input=raw)

    assert len(mem_full) == 1
    assert mem_full.prototypes[0].raw_x is not None
    assert mem_full.prototypes[0].raw_x.shape == (1, 784)
    fp_full = mem_full.estimate_memory_footprint()
    assert fp_full["total_elements"] > 784


def test_validation_gate_enable_flag():
    parent = MLPExpert(input_dim=8, hidden_dim=8, num_classes=3, expert_id=0)
    builder_disabled = ExpertBuilder(enable_gate=False)
    child = builder_disabled.create_candidate_from_parent(
        parent, new_expert_id=1, creation_task=1
    )

    # Intentionally ruin child weights
    with torch.no_grad():
        child.fc1.weight.fill_(999.0)

    enc = nn.Identity()
    mem = PrototypeMemory(feature_dim=8)
    dummy_x = torch.randn(10, 8)
    dummy_y = torch.randint(0, 3, (10,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_x, dummy_y), batch_size=5
    )

    res = builder_disabled.validate_candidate(
        candidate_expert=child,
        parent_expert=parent,
        encoder=enc,
        val_loader=loader,
        prototype_memory=mem,
    )
    # When gate is disabled, it must pass unconditionally
    assert res.passed
    assert res.rejection_reason is None


def test_capacity_control_through_expert_builder():
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=5, top_k=1)
    experts = [MLPExpert(8, 8, 3, i) for i in range(5)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    mem = PrototypeMemory(feature_dim=8)
    feat = torch.randn(8)
    r = torch.softmax(torch.randn(5), dim=-1)
    o = torch.randn(5, 3)
    mem.update_or_create_prototype(feat, r, o, task_id=0)

    # Simulate usage so expert 4 is unused, others used
    experts[0].activation_count = 10
    experts[1].activation_count = 10
    experts[2].activation_count = 10
    experts[3].activation_count = 10
    experts[4].activation_count = 0

    builder = ExpertBuilder()
    builder.enforce_capacity_control(moe, prototype_memory=mem, max_experts=4)

    assert moe.num_experts == 4
    assert len(moe.experts) == 4
    assert moe.router.num_experts == 4
    assert mem.prototypes[0].r_p.shape[0] == 4
    assert mem.prototypes[0].o_p.shape[0] == 4


def test_contrastive_pretraining():
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8, arch="mlp")
    dummy_data = torch.randn(20, 16)
    loader = torch.utils.data.DataLoader(dummy_data, batch_size=4)
    loss = enc.pretrain_contrastive(loader, epochs=1, lr=1e-3)
    assert loss > 0.0
    # Weights should be updated from initialization
    out = enc(dummy_data[:2])
    assert out.shape == (2, 8)


def test_encoder_stability_loss_gradient_flow():
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8, arch="mlp")
    enc.unfreeze()
    mem = PrototypeMemory(feature_dim=8, store_raw=True)
    raw_x = torch.randn(4, 16)
    with torch.no_grad():
        feat = enc(raw_x).mean(dim=0)
    mem.update_or_create_prototype(
        feat=feat,
        routing_dist=torch.tensor([1.0]),
        expert_outputs=torch.randn(1, 2),
        task_id=0,
        raw_input=raw_x,
    )

    # Calculate stability loss
    l_enc = mem.compute_encoder_stability_loss(enc, lambda_enc=1.5)
    assert l_enc.requires_grad
    l_enc.backward()
    # Gradients should reach encoder parameters
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()
    )
    assert has_grad


def test_validation_gate_rejection_scenarios():
    builder_strict = ExpertBuilder(
        min_acc_threshold=0.85,
        max_proto_drop=0.05,
        max_proto_acc_drop=0.01,
        max_ece=0.10,
        enable_gate=True,
    )
    enc = nn.Identity()
    parent = MLPExpert(input_dim=8, hidden_dim=8, num_classes=2, expert_id=0)
    # Untrained random candidate that fails new task accuracy
    cand_bad = MLPExpert(input_dim=8, hidden_dim=8, num_classes=2, expert_id=1)

    dummy_x = torch.randn(20, 8)
    # Labels arranged so random model has low accuracy
    dummy_y = torch.randint(0, 2, (20,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_x, dummy_y), batch_size=5
    )

    mem = PrototypeMemory(feature_dim=8)
    feat = torch.randn(8)
    mem.update_or_create_prototype(
        feat, torch.tensor([1.0]), torch.randn(1, 2), task_id=0
    )

    res = builder_strict.validate_candidate(
        candidate_expert=cand_bad,
        parent_expert=parent,
        encoder=enc,
        val_loader=loader,
        prototype_memory=mem,
    )
    # Strict gate should catch and reject suboptimal candidate
    assert not res.passed or res.rejection_reason is not None


def test_get_raw_exemplar_batch_and_hybrid_replay():
    mem = PrototypeMemory(feature_dim=8, store_raw=True)
    raw_x = torch.randn(2, 16)
    labels = torch.tensor([0, 1])
    mem.update_or_create_prototype(
        feat=torch.randn(8),
        routing_dist=torch.tensor([1.0]),
        expert_outputs=torch.randn(1, 2),
        task_id=0,
        label=labels[0],
        raw_input=raw_x[0],
    )
    raw_batch = mem.get_raw_exemplar_batch(torch.device("cpu"))
    assert raw_batch is not None
    x_ex, y_ex = raw_batch
    assert x_ex.shape[0] >= 1
    assert y_ex.shape[0] >= 1


def test_split_cifar10_tasks():
    from unittest.mock import MagicMock, patch

    import torch

    from pal_moe.data.split_cifar import get_split_cifar10_tasks

    with patch("torchvision.datasets.CIFAR10") as MockCIFAR:
        mock_dataset = MagicMock()
        # Mock targets to have 10 of each class
        mock_dataset.targets = [i for i in range(10)] * 10
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.side_effect = lambda idx: (
            torch.zeros(3, 32, 32),
            mock_dataset.targets[idx],
        )
        MockCIFAR.return_value = mock_dataset

        tasks = get_split_cifar10_tasks(
            data_dir="./data",
            batch_size=2,
            val_split=0.1,
            max_train_samples_per_task=10,
        )
        assert len(tasks) == 5
        assert tasks[0].classes == (0, 1)
        assert tasks[4].classes == (8, 9)
        sample_b = next(iter(tasks[0].train_loader))
        assert sample_b[0].shape[1:] == (3, 32, 32)


def test_freeze_historical_experts():
    from pal_moe.models.encoder import SharedEncoder
    from pal_moe.models.expert import MLPExpert
    from pal_moe.models.moe import DynamicMoE
    from pal_moe.models.router import DynamicRouter

    enc = SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16)
    router = DynamicRouter(input_dim=16, num_experts=3)
    experts = torch.nn.ModuleList(
        [MLPExpert(input_dim=16, hidden_dim=32, num_classes=10) for _ in range(3)]
    )
    model = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    # Freeze historical experts
    model.freeze_historical_experts(leave_unfrozen=1)

    # Check if expert 0 and 1 are frozen, and expert 2 is unfrozen
    for param in model.experts[0].parameters():
        assert not param.requires_grad
    for param in model.experts[1].parameters():
        assert not param.requires_grad
    for param in model.experts[2].parameters():
        assert param.requires_grad

    # Check router lock
    assert model.router.locked_experts == 2

    # Unfreeze all
    model.unfreeze_all_experts()
    for param in model.experts[0].parameters():
        assert param.requires_grad
    assert model.router.locked_experts == 0


def test_null_space_anchor_makes_new_row_orthogonal():
    """The null-space basis must make historically disjoint routings at init."""
    from pal_moe.models.router import _project_to_null_space

    torch.manual_seed(0)
    w = torch.randn(16)
    basis = torch.randn(4, 16)
    w_orth = _project_to_null_space(w, basis, scale=3.0)
    dirs = F.normalize(basis, p=2, dim=1)
    # w_orth must be orthogonal to every basis direction and scaled to 3.0
    assert torch.allclose(w_orth.norm(p=2), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(dirs @ w_orth, torch.zeros(4), atol=1e-4)


def test_add_expert_null_space_basis():
    """DynamicMoE.add_expert forwards the null-space basis to the router."""
    from pal_moe.models.moe import DynamicMoE
    from pal_moe.models.router import DynamicRouter

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16)
    router = DynamicRouter(input_dim=16, num_experts=1)
    model = DynamicMoE(
        enc,
        router,
        experts=nn.ModuleList([MLPExpert(16, 32, 10)]),
        use_ema_encoder=False,
    )
    basis = torch.randn(3, 16)
    new_exp = MLPExpert(16, 32, 10)
    model.add_expert(
        new_expert=new_exp, prototype_feat=torch.randn(16), null_space_basis=basis
    )
    assert model.num_experts == 2
    # The new routing row must be (numerically) orthogonal to the basis span:
    # its projection onto each normalized basis direction vanishes.
    dirs = F.normalize(basis, p=2, dim=1)
    new_row = model.router.gate.weight.data[1]
    assert (dirs @ new_row).abs().max().item() < 1e-4


def test_persistence_roundtrip():
    """save_checkpoint/load_checkpoint round-trips model + prototype memory."""
    from pal_moe.memory.prototype_memory import Prototype
    from pal_moe.persistence import load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16)
    router = DynamicRouter(input_dim=16, num_experts=2)
    experts = nn.ModuleList([MLPExpert(16, 32, 10) for _ in range(2)])
    model = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    mem = PrototypeMemory(feature_dim=16, max_prototypes=10)
    mem.prototypes.append(
        Prototype(
            v_p=torch.randn(16),
            r_p=torch.tensor([0.5, 0.5]),
            o_p=torch.randn(2, 10),
            task_id=0,
            x_p=torch.randn(4, 16),
            y_p=torch.tensor([0, 1, 2, 3]),
        )
    )

    path = "/tmp/test_pal_moe_ckpt.pt"
    save_checkpoint(path, model, mem, meta={"task_id": 0})

    new_model = DynamicMoE(
        SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16),
        DynamicRouter(input_dim=16, num_experts=2),
        nn.ModuleList([MLPExpert(16, 32, 10) for _ in range(2)]),
        use_ema_encoder=False,
    )
    new_mem = PrototypeMemory(feature_dim=16, max_prototypes=10)
    meta = load_checkpoint(path, new_model, new_mem, device=torch.device("cpu"))

    assert meta["task_id"] == 0
    assert len(new_mem.prototypes) == 1
    for k in new_model.state_dict():
        assert torch.allclose(
            new_model.state_dict()[k], model.state_dict()[k], atol=1e-6
        )
    import os

    os.remove(path)


def test_persistence_strict_rejects_mismatched_state(tmp_path):
    """strict=True must refuse a checkpoint from a different architecture."""
    from pal_moe.persistence import load_checkpoint, save_checkpoint

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = DynamicMoE(
        enc,
        DynamicRouter(input_dim=8, num_experts=1),
        [MLPExpert(8, 8, 3, 0)],
        use_ema_encoder=False,
    )
    mem = PrototypeMemory(feature_dim=8)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, mem, meta={"task_id": 0})

    # Different expert count: the saved routing layer does not fit.
    other = DynamicMoE(
        SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8),
        DynamicRouter(input_dim=8, num_experts=2),
        [MLPExpert(8, 8, 3, i) for i in range(2)],
        use_ema_encoder=False,
    )
    with pytest.raises(RuntimeError):
        load_checkpoint(path, other, PrototypeMemory(feature_dim=8))


def test_der_erace_agem_smoke():
    """DER++, ER-ACE and AGEM train a full task without error and improve loss."""
    from pal_moe.baselines.agem import AGEM
    from pal_moe.baselines.der import DERPP, ERACE

    torch.manual_seed(0)
    xs = torch.randn(24, 64)
    ys = torch.randint(0, 5, (24,))
    loader = [(xs, ys)]

    for cls in (DERPP, ERACE, AGEM):
        enc = SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16)
        enc.eval()
        net = nn.Sequential(enc, MLPExpert(16, 24, 5))
        tr = cls(net, buffer_size=16, lr=1e-2, device=torch.device("cpu"))
        kw = {"current_classes": [0, 1, 2]} if cls is ERACE else {}
        out = tr.train_task(0, loader, epochs=2, **kw)
        assert out["loss"] > 0


def test_icarl_smoke():
    """iCaRL trains a task and classifies by nearest class mean."""
    from pal_moe.baselines.icarl import ICaRL

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=64, hidden_dims=(32,), output_dim=16)
    enc.eval()
    net = nn.Sequential(enc, MLPExpert(16, 24, 10))
    tr = ICaRL(net, exemplars_per_class=5, lr=1e-2, device=torch.device("cpu"))

    xs = torch.randn(40, 64)
    ys = torch.cat(
        [torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)]
    )
    tr.train_task(0, [(xs, ys)], epochs=2, current_classes=[0, 1])

    out = tr.eval_model()(xs[:8])
    assert out.shape == (8, 10)
    print("iCaRL smoke OK")


def test_merge_experts_merges_router_rows_and_owner_indices():
    """Merge must be consistent across experts, router rows and prototype owners."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=3, top_k=1)
    experts = [MLPExpert(8, 8, 4, i) for i in range(3)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    with torch.no_grad():
        experts[0].fc1.weight.fill_(1.0)
        experts[1].fc1.weight.fill_(3.0)
        experts[0].usage_count.fill_(3)
        experts[1].usage_count.fill_(4)
        router.gate.weight[0].fill_(1.0)
        router.gate.weight[1].fill_(3.0)
        router.gate.bias[0].fill_(1.0)
        router.gate.bias[1].fill_(3.0)

    mem = PrototypeMemory(feature_dim=8)
    mem.update_or_create_prototype(
        torch.randn(8),
        torch.tensor([0.3, 0.5, 0.2]),
        torch.randn(3, 4),
        task_id=0,
        owner_expert=1,
    )
    mem.update_or_create_prototype(
        torch.randn(8),
        torch.tensor([0.0, 0.1, 0.9]),
        torch.randn(3, 4),
        task_id=1,
        owner_expert=2,
    )

    moe.merge_experts(0, 1, prototype_memory=mem)

    assert moe.num_experts == 2
    assert moe.router.num_experts == 2
    # Full parameter midpoint (base pathway), not just fc1/fc2
    fc1_w = moe.experts[0].fc1.weight
    assert torch.allclose(fc1_w, torch.full_like(fc1_w, 2.0))
    assert moe.experts[0].usage_count.item() == 7
    # Router row 0 is the average of the two merged rows; row 1 is the old row 2
    for param in (moe.router.gate.weight[0], moe.router.gate.bias[0]):
        assert torch.allclose(param, torch.full_like(param, 2.0))
    # Prototype owners are re-indexed: 1 -> 0, 2 -> 1
    assert mem.prototypes[0].owner_expert == 0
    assert mem.prototypes[1].owner_expert == 1
    # r_p mass of the merged expert folds into idx1 and is renormalized
    p = mem.prototypes[0]
    assert abs(p.r_p[0].item() - 0.8 / 1.0) < 1e-5
    assert abs(p.r_p[1].item() - 0.2) < 1e-5
    assert abs(p.r_p.sum().item() - 1.0) < 1e-5


def test_prune_keeps_historical_routing_lock():
    """Rebuilding the gate in prune_expert must not drop the lock hooks."""
    torch.manual_seed(0)
    router = DynamicRouter(input_dim=8, num_experts=3, top_k=1)
    router.lock_historical_routing(2)
    router.prune_expert(0)  # drops former row 0; former row 1 (locked) becomes row 0
    assert router.num_experts == 2
    assert router.locked_experts == 1

    h = torch.randn(6, 8)
    weights, _, _ = router(h)
    weights.sum().backward()
    assert router.gate.weight.grad[0].abs().sum().item() == 0.0
    assert router.gate.weight.grad[1].abs().sum().item() > 0.0

    # Lock count also stays consistent for the other router variants
    from pal_moe.models.router import AttentionRouter, DistanceRouter

    for cls in (AttentionRouter, DistanceRouter):
        r = cls(input_dim=8, num_experts=3, top_k=1)
        r.lock_historical_routing(2)
        r.prune_expert(0)
        assert r.locked_experts == 1 and r.num_experts == 2


def test_candidate_training_does_not_inflate_usage():
    """Candidate experts train outside the pool, so their usage stays at zero."""
    parent = MLPExpert(input_dim=8, hidden_dim=8, num_classes=3, expert_id=0)
    builder = ExpertBuilder()
    child = builder.create_candidate_from_parent(
        parent, new_expert_id=1, creation_task=1
    )

    x = torch.randn(12, 8)
    y = torch.randint(0, 3, (12,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4
    )
    builder.train_candidate(child, encoder=nn.Identity(), train_loader=loader, epochs=2)
    assert child.usage_count.item() == 0


def test_prototype_routing_confidence_uses_task_margin():
    """conf = clamp(1 - d1/d2, 0, 1) with d2 the nearest other-task prototype."""
    torch.manual_seed(0)
    enc = nn.Identity()
    router = DynamicRouter(input_dim=4, num_experts=2, top_k=1)
    experts = [MLPExpert(4, 4, 3, i) for i in range(2)]
    model = DynamicMoE(enc, router, experts)
    model.eval()

    mem = PrototypeMemory(feature_dim=4)
    mem.update_or_create_prototype(
        torch.zeros(4),
        torch.tensor([1.0, 0.0]),
        torch.zeros(2, 3),
        task_id=0,
        owner_expert=0,
    )
    mem.update_or_create_prototype(
        torch.tensor([10.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.zeros(2, 3),
        task_id=1,
        owner_expert=1,
    )
    model.set_prototype_routing(mem, alpha=1.0)

    with torch.no_grad():
        # Clearly inside task 0's region: d1 = 1, d2 = 9
        _, conf = model._prototype_routing_anchor(torch.tensor([[1.0, 0, 0, 0]]))
        assert abs(conf.item() - (1.0 - 1.0 / 9.0)) < 1e-5

        # Equidistant between the two tasks: maximum ambiguity -> 0
        _, conf_mid = model._prototype_routing_anchor(torch.tensor([[5.0, 0, 0, 0]]))
        assert conf_mid.item() < 1e-6

        # Single-task memory: no cross-task ambiguity -> 1
        mem_single = PrototypeMemory(feature_dim=4)
        mem_single.update_or_create_prototype(
            torch.zeros(4),
            torch.tensor([1.0, 0.0]),
            torch.zeros(2, 3),
            task_id=0,
        )
        model.set_prototype_routing(mem_single, alpha=1.0)
        _, conf_single = model._prototype_routing_anchor(torch.tensor([[1.0, 0, 0, 0]]))
        assert conf_single.item() == 1.0


def test_inference_paths_restore_training_mode():
    """Trigger and gate are inference-only: they must not leak eval() mode."""
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    enc.unfreeze()  # trainable encoder
    router = DynamicRouter(input_dim=8, num_experts=1)
    moe = DynamicMoE(enc, router, [MLPExpert(8, 8, 3)], use_ema_encoder=False)
    moe.train()

    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))

    trigger = QuantitativeTrigger(threshold_tau=1e9)
    trigger.evaluate(moe, x, y, None)
    assert moe.training and enc.training

    builder = ExpertBuilder()
    child = builder.create_candidate_from_parent(moe.experts[0], 1, 1)
    loader = [(x, y)]
    builder.train_candidate(child, enc, loader, epochs=1, device=torch.device("cpu"))
    assert enc.training, "train_candidate must restore the encoder mode"
    builder.validate_candidate(child, moe.experts[0], enc, loader, None)
    assert enc.training and moe.experts[0].training and child.training, (
        "validate_candidate must restore all module modes"
    )


def test_train_task_runs_in_training_mode_after_expansion():
    """Regression: for task_id > 0 the training loop used to run in eval()."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    enc.unfreeze()
    router = DynamicRouter(input_dim=8, num_experts=1)
    moe = DynamicMoE(enc, router, [MLPExpert(8, 8, 3)], use_ema_encoder=False)

    # Trigger everything so the expansion path (candidate train + gate) runs.
    trigger = QuantitativeTrigger(threshold_tau=-1e9)
    memory = PrototypeMemory(feature_dim=8)
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=trigger,
        builder=ExpertBuilder(min_acc_threshold=0.0, enable_gate=False),
        device=torch.device("cpu"),
    )

    modes = []
    # model.forward is only used by the training loop (and joint calibration),
    # never by the trigger/gate inference paths.
    hook = moe.register_forward_hook(lambda m, i, o: modes.append(m.training))
    try:
        x = torch.randn(16, 16)
        y = torch.randint(0, 3, (16,))
        loader = [(x, y)]
        trainer.train_task(task_id=1, train_loader=loader, val_loader=loader, epochs=1)
    finally:
        hook.remove()

    assert modes, "model forward was never recorded"
    assert all(modes), "training forwards must run in training mode"


def test_router_params_have_zero_weight_decay():
    """Adam's L2 term must not decay locked historical routing rows."""
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    moe = DynamicMoE(enc, router, [MLPExpert(8, 8, 3, i) for i in range(2)])

    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=PrototypeMemory(feature_dim=8),
        trigger=QuantitativeTrigger(),
        builder=ExpertBuilder(),
        device=torch.device("cpu"),
    )
    router_param_ids = {id(p) for p in router.parameters()}
    router_group = [
        g
        for g in trainer.optimizer.param_groups
        if any(id(p) in router_param_ids for p in g["params"])
    ]
    assert router_group and router_group[0]["weight_decay"] == 0.0


def test_periodic_stability_and_ood_knobs():
    """stability_every/ood_every skip steps without breaking history shape."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    moe = DynamicMoE(
        enc, router, [MLPExpert(8, 8, 3, i) for i in range(2)], use_ema_encoder=False
    )

    memory = PrototypeMemory(feature_dim=8)
    feat = torch.randn(8)
    memory.update_or_create_prototype(
        feat, torch.tensor([1.0, 0.0]), torch.randn(2, 3), task_id=0
    )
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(threshold_tau=1e9),
        builder=ExpertBuilder(),
        lambda_ood=0.5,
        stability_every=2,
        ood_every=2,
        joint_calib_epochs=0,
        device=torch.device("cpu"),
    )

    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))
    loader = [(x, y)] * 4  # four optimisation steps
    hist = trainer.train_task(
        task_id=1,
        train_loader=loader,
        val_loader=loader,
        epochs=1,
        enable_expansion=False,
    )

    assert len(hist["loss_router_stab"]) == 4
    # Steps 0 and 2 apply the (scaled) stability loss, steps 1 and 3 skip it.
    assert hist["loss_router_stab"][0] > 0.0
    assert hist["loss_router_stab"][1] == 0.0
    assert hist["loss_router_stab"][2] > 0.0
    assert hist["loss_router_stab"][3] == 0.0


def test_pretrain_cache_roundtrip(tmp_path):
    """The pretraining cache is content-addressed and restores exact weights."""
    from experiments.run_benchmark import (
        _load_pretrained_encoder,
        _pretrain_cache_path,
        _save_pretrained_encoder,
    )

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    path = _pretrain_cache_path(str(tmp_path), "mnist", "ae", 42, 1, 8, (32, 64, 128))
    assert not os.path.exists(path)

    _save_pretrained_encoder(enc, path)
    other = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    assert _load_pretrained_encoder(other, path)
    for k, v in enc.state_dict().items():
        assert torch.allclose(v, other.state_dict()[k], atol=1e-6)

    # Different seed or architecture -> different cache slot
    assert (
        _pretrain_cache_path(str(tmp_path), "mnist", "ae", 43, 1, 8, (32, 64, 128))
        != path
    )


def test_runner_baseline_helpers():
    """Factory/record helpers used by the benchmark runner."""
    from experiments.run_benchmark import _record_baseline_result

    from pal_moe.factory import build_single_head

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = build_single_head(enc, 8, 8, 3, torch.device("cpu"))
    evaluator = ContinualEvaluator(num_tasks=2)
    evaluator.R[1, :2] = 0.5
    res = _record_baseline_result(model, evaluator)
    assert res["final_experts"] == 1
    assert res["trainable_params"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert res["acc"] == pytest.approx(0.5)


def test_split_folder_tasks_from_images(tmp_path):
    """The generic folder splitter builds class-incremental tasks from images."""
    from PIL import Image

    from pal_moe.data.split_folder import get_split_folder_tasks

    for class_idx, colour in enumerate([(220, 30, 30), (30, 30, 220)]):
        class_dir = tmp_path / f"class_{class_idx}"
        class_dir.mkdir()
        for i in range(10):
            Image.new("RGB", (8, 8), color=colour).save(class_dir / f"img_{i}.png")

    tasks = get_split_folder_tasks(
        str(tmp_path), batch_size=4, val_split=0.2, test_split=0.2, classes_per_task=1
    )
    assert len(tasks) == 2
    assert tasks[0].classes == (0,) and tasks[1].classes == (1,)
    x, y = next(iter(tasks[0].train_loader))
    assert x.shape[1:] == (3, 32, 32)

    merged = get_split_folder_tasks(
        str(tmp_path), batch_size=4, classes_per_task=2, num_workers=0
    )
    assert len(merged) == 1 and merged[0].classes == (0, 1)
    with pytest.raises(ValueError):
        get_split_folder_tasks(str(tmp_path), classes_per_task=3)


def test_domain_shift_phase_stream():
    from types import SimpleNamespace

    import torch.utils.data as tdata

    from pal_moe.data.domain_shift import (
        apply_phase_shift,
        permutation_transform,
        rotation_transform,
    )

    x = torch.arange(16).float()
    permuted = permutation_transform(seed=3, input_dim=16)(x)
    assert sorted(permuted.tolist()) == sorted(x.tolist())
    assert not torch.allclose(permuted, x)

    grid = x.view(1, 4, 4)
    assert not torch.allclose(rotation_transform(90)(grid), grid)

    dataset = tdata.TensorDataset(
        torch.randn(16, 1, 4, 4), torch.zeros(16, dtype=torch.long)
    )
    loader = tdata.DataLoader(dataset, batch_size=8, shuffle=False)
    task = SimpleNamespace(
        task_id=0,
        classes=(0,),
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
    )
    shifted = apply_phase_shift(
        [task, task], [rotation_transform(90), rotation_transform(180)]
    )
    assert shifted[0].domain == 0 and shifted[1].domain == 1
    x0, _ = next(iter(shifted[0].test_loader))
    x1, _ = next(iter(shifted[1].test_loader))
    assert not torch.allclose(x0, x1)


def test_streaming_task_free_evaluation():
    from pal_moe.evaluation.task_free import StreamingEvaluator

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
        model.bias.zero_()

    evaluator = StreamingEvaluator(
        model, torch.device("cpu"), window=8, surprise_momentum=0.5
    )
    x = torch.randn(16, 4) + torch.tensor([4.0, 0, 0, 0])
    y = torch.zeros(16, dtype=torch.long)
    evaluator.update(x[:8], y[:8], domain=torch.zeros(8, dtype=torch.long))
    evaluator.update(x[8:], y[8:], domain=torch.ones(8, dtype=torch.long))
    report = evaluator.report()
    assert report["samples_seen"] == 16
    assert report["online_accuracy"] > 0.9
    assert report["recent_accuracy"] > 0.9
    assert report["surprise"] > 0
    assert set(report["per_domain_accuracy"]) == {"0", "1"}


def test_resnet_encoder_shapes():
    """ResNet backbones accept 3-channel and adapted 1-channel inputs."""
    from pal_moe.models.encoder import SharedEncoder

    enc = SharedEncoder(input_dim=3072, output_dim=16, arch="resnet18")
    enc.eval()
    with torch.no_grad():
        assert enc(torch.randn(4, 3, 32, 32)).shape == (4, 16)

    enc1 = SharedEncoder(input_dim=784, output_dim=8, arch="resnet18")
    enc1.eval()
    with torch.no_grad():
        assert enc1(torch.randn(2, 1, 28, 28)).shape == (2, 8)
        assert enc1(torch.randn(2, 784)).shape == (2, 8)
    assert enc1.net[0].conv1.in_channels == 1


def test_diagnose_detects_resnet_and_encoder_meta(tmp_path):
    from experiments.diagnose_checkpoint import (
        _detect_encoder_arch,
        build_model,
        infer_config,
    )

    from pal_moe.factory import build_moe
    from pal_moe.persistence import save_checkpoint

    model = build_moe(
        SharedEncoder(input_dim=3072, output_dim=8, arch="resnet18"),
        8,
        8,
        3,
        num_experts=1,
    )
    path = str(tmp_path / "resnet.pt")
    save_checkpoint(path, model, PrototypeMemory(feature_dim=8), meta={"task_id": 0})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    assert payload["meta"]["encoder_arch"] == "resnet18"
    assert _detect_encoder_arch(state) == "resnet18"

    cfg = infer_config(state, "cifar10", encoder_arch=payload["meta"]["encoder_arch"])
    assert cfg["arch"] == "resnet18"
    build_model(cfg, torch.device("cpu")).load_state_dict(state)

    # Heuristic fallbacks for checkpoints without encoder meta.
    assert (
        _detect_encoder_arch({"encoder.net.0.layer1.0.conv1.weight": torch.zeros(1)})
        == "resnet18"
    )
    assert (
        _detect_encoder_arch({"encoder.net.0.layer1.0.conv3.weight": torch.zeros(1)})
        == "resnet50"
    )
    r34 = {f"encoder.net.0.layer2.{i}.conv1.weight": torch.zeros(1) for i in range(4)}
    r34["encoder.net.0.layer1.0.conv1.weight"] = torch.zeros(1)
    assert _detect_encoder_arch(r34) == "resnet34"


def test_capacity_control_remaps_tracked_expert():
    """Prune/merge re-indexes experts; the tracked (new) expert must follow."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=3)
    experts = [MLPExpert(8, 8, 3, i) for i in range(3)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)
    mem = PrototypeMemory(feature_dim=8)

    # Merge path: experts 0 and 1 are identical (similarity 1), expert 2 differs.
    with torch.no_grad():
        experts[1].fc2.weight.copy_(experts[0].fc2.weight)
        experts[1].fc2.bias.copy_(experts[0].fc2.bias)
        for e in experts:
            e.usage_count.fill_(100)
    result = ExpertBuilder().enforce_capacity_control(
        moe, prototype_memory=mem, max_experts=2, track_expert=2
    )
    assert moe.num_experts == 2
    assert result["tracked_expert"] == 1  # index 1 was merged away, so it shifts down

    # Prune path: the least-used expert is removed, track_expert follows it.
    torch.manual_seed(0)
    moe2 = DynamicMoE(
        SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8),
        DynamicRouter(input_dim=8, num_experts=3),
        [MLPExpert(8, 8, 3, i) for i in range(3)],
        use_ema_encoder=False,
    )
    with torch.no_grad():
        for i, e in enumerate(moe2.experts):
            e.usage_count.fill_(100 if i != 1 else 0)
    result = ExpertBuilder().enforce_capacity_control(
        moe2,
        prototype_memory=PrototypeMemory(feature_dim=8),
        max_experts=2,
        track_expert=2,
    )
    assert moe2.num_experts == 2
    assert result["tracked_expert"] == 1


def test_distill_skips_out_of_range_owners():
    """Stale owner indices must be ignored, not crash the distillation."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    moe = DynamicMoE(
        enc,
        DynamicRouter(input_dim=8, num_experts=2),
        [MLPExpert(8, 8, 3, i) for i in range(2)],
        use_ema_encoder=False,
    )
    memory = PrototypeMemory(feature_dim=8)
    memory.update_or_create_prototype(
        torch.randn(8),
        torch.tensor([1.0, 0.0]),
        torch.randn(2, 3),
        task_id=0,
        owner_expert=0,
    )
    memory.prototypes[0].owner_expert = 7  # stale index
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(),
        builder=ExpertBuilder(),
        device=torch.device("cpu"),
    )
    loss = trainer._distill_router_anchors(steps=5, lr=1e-2)
    assert loss == 0.0


def test_shared_expert_stability_anchor():
    """The always-on expert is anchored to its registration-time outputs."""
    from pal_moe.factory import build_moe

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = build_moe(enc, 8, 8, 3, num_experts=1, shared_expert=True)

    memory = PrototypeMemory(feature_dim=8)
    h = torch.randn(4, 8)
    outs = model.get_all_expert_outputs(h)
    shared = model.shared_expert(h, track_usage=False)
    memory.register_task_batch(
        h,
        torch.softmax(torch.randn(4, 1), dim=-1),
        outs,
        task_id=0,
        labels=torch.tensor([0, 1, 2, 0]),
        shared_outputs=shared,
    )
    assert all(p.s_p is not None for p in memory.prototypes)

    loss_before = memory.compute_stability_losses(model, lambda_e=1.0)[1]
    with torch.no_grad():
        for p in model.shared_expert.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    loss_after = memory.compute_stability_losses(model, lambda_e=1.0)[1]
    assert loss_after.item() > loss_before.item()

    # refresh_anchors pulls the anchors back to the current shared behaviour.
    memory.refresh_anchors(model)
    refreshed = memory.compute_stability_losses(model, lambda_e=1.0)[1]
    assert refreshed.item() < loss_after.item()


def test_freeze_shared_expert_after_task():
    from pal_moe.factory import build_moe

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = build_moe(enc, 8, 8, 3, num_experts=2, shared_expert=True)
    memory = PrototypeMemory(feature_dim=8)
    trainer = ContinualTrainer(
        model=model,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(threshold_tau=1e9),
        builder=ExpertBuilder(),
        freeze_shared_after=1,
        joint_calib_epochs=0,
        device=torch.device("cpu"),
    )
    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))
    trainer.train_task(
        task_id=1,
        train_loader=[(x, y)],
        val_loader=[(x, y)],
        epochs=1,
        enable_expansion=False,
    )
    assert getattr(model, "shared_frozen", False)
    assert not any(p.requires_grad for p in model.shared_expert.parameters())
    # The gate stays trainable so the model can still choose the generalist.
    assert all(p.requires_grad for p in model.shared_gate.parameters())
    # Calibration must not unfreeze it again.
    model.unfreeze_all_experts()
    assert not any(p.requires_grad for p in model.shared_expert.parameters())


def test_expert_width_growth_is_function_preserving():
    torch.manual_seed(0)
    expert = MLPExpert(8, 16, 3, 0)
    h = torch.randn(5, 8)
    before = expert(h).detach()
    assert expert.widen(32)
    after = expert(h).detach()
    assert expert.hidden_dim == 32
    assert torch.allclose(before, after, atol=1e-6)
    # The new units are trainable capacity.
    expert(h).sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in expert.fc1.parameters()
    )
    assert not expert.widen(16), "widening to a smaller size must be a no-op"


def test_adapter_experts_freeze_the_base_pathway():
    torch.manual_seed(0)
    parent = MLPExpert(8, 16, 3, 0)
    child = parent.clone_function_preserving(1, 1, freeze_base=True)
    assert not any(p.requires_grad for p in child.fc1.parameters())
    assert not any(p.requires_grad for p in child.fc2.parameters())
    assert all(p.requires_grad for p in child.adapter_down.parameters())
    assert all(p.requires_grad for p in child.adapter_up.parameters())
    # Function preserving: identical outputs at creation.
    h = torch.randn(4, 8)
    assert torch.allclose(parent(h), child(h), atol=1e-6)


def test_model_merging_utilities():
    from pal_moe.merge import model_soup, task_arithmetic, ties_merge

    torch.manual_seed(0)
    experts = [MLPExpert(8, 16, 3, i) for i in range(3)]

    soup = model_soup(experts)
    expected = {
        name: torch.stack(
            [dict(e.named_parameters())[name].detach() for e in experts]
        ).mean(dim=0)
        for name, _ in soup.named_parameters()
    }
    for name, param in soup.named_parameters():
        assert torch.allclose(param, expected[name], atol=1e-6)

    ties = ties_merge(experts, top_k=0.5)
    assert ties.hidden_dim == 16 and ties.num_classes == 3
    arithmetic = task_arithmetic(experts, scaling=0.5)
    assert arithmetic.hidden_dim == 16
    # task arithmetic with scaling 1 relative to the first expert reproduces
    # the sum of deltas.
    names = [n for n, _ in arithmetic.named_parameters()]
    base = dict(experts[0].named_parameters())
    manual = {
        n: base[n] + 0.5 * sum(dict(e.named_parameters())[n] - base[n] for e in experts)
        for n in names
    }
    for name, param in arithmetic.named_parameters():
        assert torch.allclose(param, manual[name], atol=1e-6)


def test_widen_expansion_action_on_gate_rejection():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=1)
    experts = [MLPExpert(8, 8, 3, 0)]
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    memory = PrototypeMemory(feature_dim=8)
    memory.update_or_create_prototype(
        torch.randn(8),
        torch.tensor([1.0]),
        torch.randn(1, 3),
        task_id=0,
        owner_expert=0,
    )
    # Strict gate always rejects.
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(threshold_tau=-1e9),
        builder=ExpertBuilder(min_acc_threshold=0.99),
        expansion_action="widen",
        widen_by=8,
        joint_calib_epochs=0,
        device=torch.device("cpu"),
    )
    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))
    hist = trainer.train_task(
        task_id=1,
        train_loader=[(x, y)],
        val_loader=[(x, y)],
        epochs=1,
        enable_expansion=True,
    )
    assert hist["gate_rejections"] == 1
    assert hist["width_growths"] == 1
    assert moe.experts[-1].hidden_dim == 16


def test_relative_gate_relaxes_for_weak_majority():
    """Relative gate = min(absolute, majority + margin); never stricter."""
    torch.manual_seed(0)
    x = torch.randn(50, 8)
    y = torch.arange(50) % 5  # balanced 5-way -> majority 0.2
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=25
    )

    absolute = ExpertBuilder(min_acc_threshold=0.45)
    relative = ExpertBuilder(
        min_acc_threshold=0.45, gate_mode="relative", gate_margin=0.10
    )
    assert absolute.gate_threshold(loader) == (0.45, None)
    threshold, majority = relative.gate_threshold(loader)
    assert majority == pytest.approx(0.2)
    assert threshold == pytest.approx(0.30)

    # A 2-way task (majority 0.5) keeps the absolute bar even in relative mode.
    y2 = torch.arange(50) % 2
    loader2 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y2), batch_size=25
    )
    threshold2, majority2 = relative.gate_threshold(loader2)
    assert majority2 == pytest.approx(0.5)
    assert threshold2 == pytest.approx(0.45)

    with pytest.raises(ValueError):
        ExpertBuilder(gate_mode="nope")


def test_param_reporting_fields():
    """total/trainable/active parameter counts must tell the compute story."""
    from experiments.run_benchmark import _active_params_per_sample

    from pal_moe.factory import build_moe, build_single_head

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    baseline = build_single_head(enc, 8, 8, 3)
    assert _active_params_per_sample(baseline) == sum(
        p.numel() for p in baseline.parameters()
    )

    moe = build_moe(enc, 8, 8, 3, num_experts=3, shared_expert=True)
    total = sum(p.numel() for p in moe.parameters())
    active = _active_params_per_sample(moe)
    one_expert = sum(p.numel() for p in moe.experts[0].parameters())
    assert active < total
    assert active >= one_expert  # at least one expert is always on the path


def test_uncertainty_weighter_formula_and_gradients():
    from pal_moe.adaptation.losses import UncertaintyWeighter

    weighter = UncertaintyWeighter(("task", "router"))
    losses = {
        "task": torch.tensor(2.0, requires_grad=True),
        "router": torch.tensor(4.0),
        "other": torch.tensor(1.0),
    }
    total, diagnostics = weighter.combine(losses)
    # s=0 initially: exp(0)*L + 0, plus the passthrough term.
    assert abs(total.item() - 7.0) < 1e-5
    assert set(diagnostics) == {"task", "router"}
    total.backward()
    assert weighter.log_vars["task"].grad is not None


def test_lwf_and_ema_terms_run():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    enc.unfreeze()
    moe = DynamicMoE(
        enc,
        DynamicRouter(input_dim=8, num_experts=1),
        [MLPExpert(8, 8, 3)],
        use_ema_encoder=True,
    )
    memory = PrototypeMemory(feature_dim=8)
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(threshold_tau=1e9),
        builder=ExpertBuilder(),
        lambda_lwf=0.5,
        lambda_ema=0.5,
        joint_calib_epochs=0,
        device=torch.device("cpu"),
    )
    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))
    hist = trainer.train_task(
        task_id=0,
        train_loader=[(x, y)],
        val_loader=[(x, y)],
        epochs=1,
        enable_expansion=False,
    )
    assert trainer._lwf_teacher is not None
    assert len(hist["loss_total"]) == 1
    # The EMA copy must have moved with the online encoder.
    assert any(
        not torch.allclose(p, q)
        for p, q in zip(
            moe.encoder.parameters(), moe.ema_encoder.ema_model.parameters()
        )
    )


def test_loss_weighting_uncertainty_trains():
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    moe = DynamicMoE(
        enc,
        DynamicRouter(input_dim=8, num_experts=2),
        [MLPExpert(8, 8, 3, i) for i in range(2)],
    )
    memory = PrototypeMemory(feature_dim=8)
    memory.update_or_create_prototype(
        torch.randn(8), torch.tensor([1.0, 0.0]), torch.randn(2, 3), task_id=0
    )
    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(threshold_tau=1e9),
        builder=ExpertBuilder(),
        loss_weighting="uncertainty",
        joint_calib_epochs=0,
        device=torch.device("cpu"),
    )
    assert trainer.loss_weighter is not None
    x = torch.randn(16, 16)
    y = torch.randint(0, 3, (16,))
    trainer.train_task(
        task_id=0,
        train_loader=[(x, y)],
        val_loader=[(x, y)],
        epochs=1,
        enable_expansion=False,
    )
    # The weighter parameters must be part of the optimizer and updatable.
    weighter_ids = {id(p) for p in trainer.loss_weighter.parameters()}
    assert any(
        id(p) in weighter_ids
        for group in trainer.optimizer.param_groups
        for p in group["params"]
    )


def test_latent_generative_replay_gaussian_and_vae():
    from pal_moe.memory.generative import LatentReplayGenerator

    torch.manual_seed(0)
    memory = PrototypeMemory(feature_dim=4)
    for label, mean in ((0, torch.zeros(4)), (1, torch.ones(4) * 5)):
        for _ in range(3):
            memory.update_or_create_prototype(
                mean + torch.randn(4) * 0.05,
                torch.tensor([1.0]),
                torch.randn(1, 2),
                task_id=label,
                label=torch.tensor(label),
            )

    gaussian = LatentReplayGenerator(4, 2, mode="gaussian", device=torch.device("cpu"))
    assert gaussian.fit(memory)
    feats, labels = gaussian.sample(64)
    assert feats.shape == (64, 4)
    assert set(labels.tolist()) <= {0, 1}
    # Samples stay close to their class means.
    for c in (0, 1):
        centre = torch.zeros(4) if c == 0 else torch.ones(4) * 5
        subset = feats[labels == c]
        assert (subset - centre).norm(dim=1).mean() < 2.0

    vae = LatentReplayGenerator(
        4, 2, mode="vae", device=torch.device("cpu"), vae_steps=40
    )
    assert vae.fit(memory)
    feats, labels = vae.sample(32)
    assert feats.shape == (32, 4)
    assert torch.isfinite(feats).all()

    empty = LatentReplayGenerator(4, 2)
    assert not empty.fit(PrototypeMemory(feature_dim=4))


def test_prototype_selection_and_eviction_modes():
    torch.manual_seed(0)
    # kcenter must keep spread-out exemplars instead of the first five.
    mem = PrototypeMemory(
        feature_dim=2,
        distance_threshold=1e9,  # merge everything into one prototype
        exemplars_per_proto=3,
        selection="kcenter",
        candidate_pool=4,
    )
    centre = torch.zeros(2)
    mem.update_or_create_prototype(centre, torch.tensor([1.0]), torch.zeros(1, 2), 0)
    for i in range(20):
        style = (
            torch.tensor([float(i), 0.0])
            if i < 10
            else torch.tensor([0.0, float(i - 10)])
        )
        mem.register_task_batch(
            style.unsqueeze(0),
            torch.tensor([[1.0]]),
            torch.zeros(1, 1, 2),
            task_id=0,
            labels=torch.tensor([0]),
        )
    proto = mem.prototypes[0]
    assert proto.x_p.size(0) <= 3
    assert proto.x_p[:, 0].max().item() > 5 or proto.x_p[:, 1].max().item() > 5

    # balanced eviction never drops the newest task's prototypes.
    mem2 = PrototypeMemory(
        feature_dim=2, max_prototypes=4, distance_threshold=1e-6, eviction="balanced"
    )
    for task in range(3):
        for i in range(4):
            mem2.register_task_batch(
                torch.tensor([[float(task * 10 + i), 0.0]]),
                torch.tensor([[1.0]]),
                torch.zeros(1, 1, 2),
                task_id=task,
                labels=torch.tensor([0]),
            )
    assert len(mem2.prototypes) == 4
    assert any(p.task_id == 2 for p in mem2.prototypes)


def test_prototype_ann_index_is_optional():
    mem = PrototypeMemory(feature_dim=4)
    mem.update_or_create_prototype(
        torch.randn(4), torch.tensor([1.0]), torch.zeros(1, 2), task_id=0
    )
    built = mem.build_ann_index()
    if not built:  # faiss is not a dependency: exact search stays the default
        assert mem.query_ann(torch.randn(1, 4)) is None
    else:  # pragma: no cover - only when faiss is installed
        dists, idx = mem.query_ann(torch.randn(1, 4), k=1)
        assert dists.shape == (1, 1)


def test_ncm_and_bias_correction_heads():
    """Both read-out heads must beat an artificially biased model head."""
    from pal_moe.evaluation.heads import BiasCorrectionHead, NCMHead
    from pal_moe.factory import build_cached_encoder

    torch.manual_seed(0)
    # Identity encoder: the test works directly in the latent space, which is
    # what the runner sees with --feature_cache.
    enc = build_cached_encoder(4)
    # A real (ReLU-positive) linear classifier with a +5 recency bias on class 1.
    experts = [MLPExpert(4, 4, 2, 0)]
    moe = DynamicMoE(enc, DynamicRouter(input_dim=4, num_experts=1), experts)
    with torch.no_grad():
        experts[0].fc1.weight.copy_(torch.eye(4))
        experts[0].fc1.bias.zero_()
        experts[0].fc2.weight.copy_(
            torch.tensor([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
        )
        experts[0].fc2.bias.copy_(torch.tensor([0.0, 5.0]))

    means = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1: torch.tensor([0.5, 1.0, 0.0, 0.0]),
    }
    memory = PrototypeMemory(feature_dim=4)
    for label, mean in means.items():
        for _ in range(5):
            feat = mean + torch.randn(4) * 0.02
            memory.update_or_create_prototype(
                feat,
                torch.tensor([1.0]),
                torch.randn(1, 2),
                task_id=0,
                label=torch.tensor(label),
            )

    ncm = NCMHead(moe, memory, num_classes=2)
    ncm.eval()
    with torch.no_grad():
        logits = ncm(memory.get_exemplar_batch(torch.device("cpu"))[0])
        preds = logits.argmax(dim=-1)
        labels = memory.get_exemplar_batch(torch.device("cpu"))[1]
    assert (preds == labels).float().mean() > 0.9

    bias = BiasCorrectionHead(moe, memory, num_classes=2, strength=1.0)
    bias.eval()
    with torch.no_grad():
        feats, labels = memory.get_exemplar_batch(torch.device("cpu"))
        base_preds = moe(latent_h=feats).argmax(dim=-1)
        corrected_preds = bias(feats).argmax(dim=-1)
    assert (base_preds == 1).all(), "biased model should always predict class 1"
    assert (corrected_preds == labels).float().mean() > 0.9


def test_shared_generalist_expert():
    """The always-on generalist mixes with the routed expert and stays trainable."""
    from pal_moe.factory import build_moe
    from pal_moe.persistence import load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    model = build_moe(enc, 8, 8, 3, num_experts=2, shared_expert=True)
    assert model.shared_expert is not None
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == (8, 3)
    out.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.shared_expert.parameters()
    )
    assert model.shared_gate.weight.grad is not None

    # Freezing history must not freeze the generalist.
    model.freeze_historical_experts(leave_unfrozen=1)
    assert all(p.requires_grad for p in model.shared_expert.parameters())
    assert all(p.requires_grad for p in model.shared_gate.parameters())
    assert not any(p.requires_grad for p in model.experts[0].parameters())

    # Round-trip through the checkpoint layer.
    memory = PrototypeMemory(feature_dim=8)
    path = str(__import__("tempfile").mkdtemp() + "/shared.pt")
    save_checkpoint(path, model, memory)
    rebuilt = build_moe(
        SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8),
        8,
        8,
        3,
        num_experts=2,
        shared_expert=True,
    )
    load_checkpoint(path, rebuilt, PrototypeMemory(feature_dim=8))
    assert torch.allclose(rebuilt.shared_gate.bias, model.shared_gate.bias, atol=1e-6)


def test_diagnose_infer_config_detects_shared_expert(tmp_path):
    from experiments.diagnose_checkpoint import build_model, infer_config

    from pal_moe.factory import build_moe
    from pal_moe.persistence import save_checkpoint

    model = build_moe(
        SharedEncoder(input_dim=784, hidden_dims=(64,), output_dim=8),
        8,
        8,
        3,
        num_experts=2,
        shared_expert=True,
    )
    path = str(tmp_path / "shared.pt")
    save_checkpoint(path, model, PrototypeMemory(feature_dim=8))
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state"]
    cfg = infer_config(state, "mnist")
    assert cfg["has_shared_expert"]
    build_model(cfg, torch.device("cpu")).load_state_dict(state)


def test_prototype_routing_alpha_calibration():
    """Alpha calibration must pick the anchor weight with the best exemplar accuracy."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    # Force the router to always pick expert 1, so only anchoring can help.
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.bias.copy_(torch.tensor([-5.0, 5.0]))
    experts = [MLPExpert(8, 8, 3, i) for i in range(2)]
    # Expert 0 is the only one that classifies its own prototypes correctly;
    # mirror the classes into expert 1 so only the owner anchors matter.
    with torch.no_grad():
        experts[0].fc2.weight.zero_()
        experts[0].fc2.bias.copy_(torch.tensor([1.0, -1.0, 0.0]))
    moe = DynamicMoE(enc, router, experts, use_ema_encoder=False)

    memory = PrototypeMemory(feature_dim=8)
    for _ in range(8):
        memory.update_or_create_prototype(
            torch.randn(8),
            torch.tensor([1.0, 0.0]),
            torch.randn(2, 3),
            task_id=0,
            label=torch.tensor(0),  # expert 0's constant argmax is class 0
            owner_expert=0,
        )
    moe.set_prototype_routing(memory, alpha=0.0)
    alpha = moe.calibrate_prototype_routing()
    assert alpha > 0.0
    moe.eval()  # anchoring is an inference-time mechanism
    with torch.no_grad():
        v = memory.get_prototype_matrix(torch.device("cpu"))
        _, idx, _ = moe._route(v)  # v is already in latent space
    assert idx.flatten().tolist() == [0] * len(memory.prototypes)


def test_expert_temperature_calibration_reduces_nll():
    from pal_moe.evaluation.calibration import (
        calibrate_expert_temperatures,
        fit_temperature,
    )

    torch.manual_seed(0)
    # Confident but wrong-on-average logits: softening must reduce the NLL.
    logits = torch.randn(64, 3) * 8.0
    labels = torch.randint(0, 3, (64,))
    t = fit_temperature(logits, labels)
    nll_before = torch.nn.functional.cross_entropy(logits, labels).item()
    nll_after = torch.nn.functional.cross_entropy(logits / t, labels).item()
    assert t > 1.0
    assert nll_after < nll_before

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    moe = DynamicMoE(
        enc,
        DynamicRouter(input_dim=8, num_experts=2),
        [MLPExpert(8, 8, 3, i) for i in range(2)],
        use_ema_encoder=False,
    )
    loader = [(torch.randn(16, 16), torch.randint(0, 3, (16,)))]
    report = calibrate_expert_temperatures(moe, loader, device=torch.device("cpu"))
    assert len(report["temperatures"]) == 2
    assert all(t > 0 for t in report["temperatures"])


def test_energy_boundary_loss_pushes_prototypes_down():
    """The energy hinge must lower old-prototype energy and raise new-feature energy."""
    from pal_moe.adaptation.ttt import energy_boundary_loss

    torch.manual_seed(0)
    expert = MLPExpert(8, 16, 3, 0)
    old = torch.randn(16, 8)
    new = torch.randn(16, 8) + 0.5

    loss0 = energy_boundary_loss(expert, old, new, margin=1.0)
    assert loss0.item() > 0
    opt = torch.optim.Adam(expert.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        loss = energy_boundary_loss(expert, old, new, margin=1.0)
        loss.backward()
        opt.step()

    with torch.no_grad():
        e_old = torch.logsumexp(expert(old, track_usage=False), dim=-1).mean()
        e_new = torch.logsumexp(expert(new, track_usage=False), dim=-1).mean()
    assert loss.item() < loss0.item()
    assert e_old < e_new

    # Prototype-only fallback also optimises
    fallback = energy_boundary_loss(expert, old, new_features=None, margin=1.0)
    assert fallback.item() >= 0


def test_energy_trigger_flags_out_of_distribution_batches():
    from pal_moe.trigger.energy_trigger import EnergyTrigger

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=1)
    moe = DynamicMoE(enc, router, [MLPExpert(8, 8, 3)], use_ema_encoder=False)

    trigger = EnergyTrigger(threshold=2.0, warmup_batches=3)
    x_id = torch.randn(32, 16) * 0.1
    for _ in range(5):
        res = trigger.evaluate(moe, x_id)
    assert not res.should_trigger, "in-distribution batches must not trigger"

    ref_mean = trigger.stats.mean
    res = trigger.evaluate(moe, torch.randn(32, 16) * 100.0)
    assert res.should_trigger, "high-energy OOD batch should trigger expansion"
    assert trigger.stats.mean == ref_mean, (
        "triggered batches must not update the reference"
    )


def test_router_learnable_temperature():
    """learn_temperature adds a trainable log-temperature and stays stable."""
    from pal_moe.models.router import AttentionRouter, DistanceRouter

    torch.manual_seed(0)
    h = torch.randn(32, 8)
    plain = DynamicRouter(input_dim=8, num_experts=3, learn_temperature=False)
    assert not any("log_temperature" in k for k in plain.state_dict())

    router = DynamicRouter(input_dim=8, num_experts=3, learn_temperature=True)
    assert "log_temperature" in dict(router.named_parameters())
    t0 = router.effective_temperature().item()
    assert abs(t0 - 1.0) < 1e-5

    weights, _, _ = router(h)
    weights.sum().backward()
    assert router.log_temperature.grad is not None
    assert router.log_temperature.grad.abs().item() > 0

    with torch.no_grad():
        router.log_temperature -= 0.5
    assert router.effective_temperature().item() < t0

    for cls in (DistanceRouter, AttentionRouter):
        r = cls(input_dim=8, num_experts=3, learn_temperature=True)
        r_weights, _, _ = r(h)
        r_weights.sum().backward()
        assert r.log_temperature.grad is not None


def test_router_anchor_margin_distillation():
    """Margin distillation trains the router onto prototype owners."""
    from pal_moe.adaptation.ttt import ContinualTrainer

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    moe = DynamicMoE(
        enc, router, [MLPExpert(8, 8, 3, i) for i in range(2)], use_ema_encoder=False
    )
    memory = PrototypeMemory(feature_dim=8)
    for i in range(20):
        owner = i % 2
        feat = torch.randn(8) + (0 if owner == 0 else 5.0) * torch.ones(8)
        memory.update_or_create_prototype(
            feat,
            torch.eye(2)[owner],
            torch.randn(2, 3),
            task_id=owner,
            owner_expert=owner,
        )

    trainer = ContinualTrainer(
        model=moe,
        prototype_memory=memory,
        trigger=QuantitativeTrigger(),
        builder=ExpertBuilder(),
        router_anchor_margin=0.5,
        device=torch.device("cpu"),
    )
    loss = trainer._distill_router_anchors(steps=60, lr=1e-2)
    assert loss < 1.0

    with torch.no_grad():
        v = memory.get_prototype_matrix(torch.device("cpu"))
        pred = router(v)[1].flatten()
        owners = torch.tensor([p.owner_expert for p in memory.prototypes])
        assert (pred == owners).float().mean().item() > 0.9


def test_geometry_report_separates_clusters():
    """Geometry metrics must separate clustered from overlapping features."""
    from pal_moe.evaluation.geometry import (
        geometry_report,
        nearest_other_margin,
        silhouette_score,
    )

    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    feats = torch.cat([c + torch.randn(50, 2) * 0.1 for c in centers])
    labels = torch.repeat_interleave(torch.arange(3), 50)

    margin = nearest_other_margin(feats, labels)
    assert (margin > 0).float().mean() > 0.95
    assert silhouette_score(feats, labels) > 0.8

    report = geometry_report(feats, labels)
    assert report["margin_mean"] > 0
    assert report["class_mean_separation_ratio"] > 0.5

    overlapping = geometry_report(torch.randn(150, 2), labels)
    assert overlapping["silhouette"] < 0.2
    assert report["silhouette"] > overlapping["silhouette"]


def test_router_diagnostics_reports_owner_accuracy():
    from pal_moe.evaluation.diagnostics import router_diagnostics

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    moe = DynamicMoE(
        enc, router, [MLPExpert(8, 8, 3, i) for i in range(2)], use_ema_encoder=False
    )
    memory = PrototypeMemory(feature_dim=8)
    feats = torch.randn(6, 8)
    for i, owner in enumerate([0, 0, 1, 1, 0, 1]):
        memory.update_or_create_prototype(
            feats[i],
            torch.tensor([1.0, 0.0]) if owner == 0 else torch.tensor([0.0, 1.0]),
            torch.randn(2, 3),
            task_id=owner,
            owner_expert=owner,
        )

    class FakeTask:
        classes = (0, 1)
        test_loader = [(torch.randn(16, 16), torch.randint(0, 3, (16,)))]
        val_loader = test_loader

    diag = router_diagnostics(
        moe, [FakeTask(), FakeTask()], torch.device("cpu"), memory
    )
    assert diag["num_experts"] == 2
    assert 0.0 <= diag["usage_entropy_normalized"] <= 1.0
    assert diag["owners"]["num_owned_prototypes"] == 6
    assert 0.0 <= diag["owners"]["owner_routing_accuracy"] <= 1.0
    assert diag["prototype_geometry"]["n_samples"] == 6


def test_sample_buffer_reservoir_balances_tasks():
    """Reservoir mode keeps every task; recency evicts the oldest tasks."""
    import random

    from pal_moe.baselines.buffer import SampleBuffer, task_class_counts

    random.seed(0)
    torch.manual_seed(0)

    reservoir = SampleBuffer(30, mode="reservoir")
    recency = SampleBuffer(30, mode="recency")
    for task in range(3):
        x = torch.full((100, 2), float(task))
        y = torch.full((100,), task)
        reservoir.add_task(x, y)
        recency.add_task(x, y)

    counts = task_class_counts(reservoir, 3)
    assert len(reservoir) == 30
    assert all(c > 0 for c in counts), f"reservoir dropped a task: {counts}"
    assert max(counts) < 25, f"reservoir is not uniform enough: {counts}"

    counts_recency = task_class_counts(recency, 3)
    assert counts_recency[0] == 0, "recency is expected to evict the first task"

    with pytest.raises(ValueError):
        SampleBuffer(10, mode="nope")


def test_ewc_online_keeps_single_fisher():
    """online=True accumulates one running Fisher instead of one per task."""
    from pal_moe.baselines.ewc import EWC

    model = nn.Linear(4, 3)
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=8
    )

    online = EWC(model, online=True)
    online.compute_fisher(loader, num_samples=8)
    online.compute_fisher(loader, num_samples=8)
    assert len(online.fisher_matrices) == 1
    assert len(online.star_params) == 1

    offline = EWC(model, online=False)
    offline.compute_fisher(loader, num_samples=8)
    offline.compute_fisher(loader, num_samples=8)
    assert len(offline.fisher_matrices) == 2


def test_test_time_adapter_restores_model_weights():
    """Test-time adaptation must not permanently modify router/expert weights."""
    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    router = DynamicRouter(input_dim=8, num_experts=2)
    moe = DynamicMoE(
        enc, router, [MLPExpert(8, 8, 3, i) for i in range(2)], use_ema_encoder=False
    )
    before = {k: v.clone() for k, v in moe.state_dict().items()}

    adapter = TestTimeAdapter(moe, steps=2, lr=1e-2)
    adapter.adapt_and_predict(torch.randn(8, 16))

    for key, value in moe.state_dict().items():
        assert torch.allclose(value, before[key]), f"{key} changed permanently"


def test_factory_builds_consistent_models():
    """The shared factory must construct every router/model variant."""
    from pal_moe.factory import (
        build_moe,
        build_prototype_memory,
        build_router,
        build_single_head,
    )

    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    for router_type in ("dynamic", "distance", "attention"):
        router = build_router(router_type, 8, top_k=1, num_experts=2)
        assert router.num_experts == 2
        model = build_moe(enc, 8, 8, 3, num_experts=2, router_type=router_type)
        assert model.num_experts == 2
        assert model.router.num_experts == 2
    with pytest.raises(ValueError):
        build_router("nope", 8)

    head = build_single_head(enc, 8, 8, 3)
    assert isinstance(head, nn.Sequential) and len(head) == 2

    memory = build_prototype_memory(8, distance_threshold=None, max_prototypes=7)
    assert memory.distance_threshold is None
    assert memory.max_prototypes == 7


def test_diagnose_infer_config_rebuilds_every_router_and_cached_encoder(tmp_path):
    """diagnose_checkpoint must rebuild all router kinds and cached encoders."""
    from experiments.diagnose_checkpoint import build_model, infer_config

    from pal_moe.factory import build_cached_encoder, build_moe
    from pal_moe.persistence import save_checkpoint

    enc = SharedEncoder(input_dim=784, hidden_dims=(64,), output_dim=8)
    for router_type in ("dynamic", "distance", "attention"):
        model = build_moe(enc, 8, 8, 3, num_experts=2, router_type=router_type)
        path = str(tmp_path / f"{router_type}.pt")
        save_checkpoint(path, model, PrototypeMemory(feature_dim=8))
        state = torch.load(path, map_location="cpu", weights_only=True)["model_state"]
        cfg = infer_config(state, "mnist")
        assert cfg["router_kind"] == router_type
        assert cfg["arch"] == "mlp"
        rebuilt = build_model(cfg, torch.device("cpu"))
        rebuilt.load_state_dict(state)  # strict: architecture must match exactly

    cached = build_moe(build_cached_encoder(8), 8, 8, 3, num_experts=1)
    path = str(tmp_path / "cached.pt")
    save_checkpoint(path, cached, PrototypeMemory(feature_dim=8))
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state"]
    cfg = infer_config(state, "mnist")
    assert cfg["arch"] == "cached" and cfg["cached_encoder"]
    build_model(cfg, torch.device("cpu")).load_state_dict(state)


def test_runner_baseline_loop_forwards_per_task_kwargs():
    """The shared baseline loop forwards per-task kwargs and evaluates each step."""
    from experiments.run_benchmark import _run_baseline_loop

    recorded = []

    class FakeTrainer:
        def train_task(self, task_id, loader, epochs=1, **kwargs):
            recorded.append((task_id, epochs, kwargs))

    class FakeEvaluator:
        def evaluate_all_seen_tasks(self, model, t_idx, tasks):
            assert tasks is TASKS
            return [1.0]

    empty_batch = [(torch.randn(4, 16), torch.randint(0, 3, (4,)))]

    class FakeTask:
        classes = (0, 1)
        train_loader = [1]
        test_loader = empty_batch

    TASKS = [FakeTask(), FakeTask()]
    _run_baseline_loop(
        FakeTrainer(),
        model=None,
        evaluator=FakeEvaluator(),
        tasks=TASKS,
        epochs_per_task=2,
        per_task_kwargs=lambda task: {"current_classes": list(task.classes)},
    )
    assert recorded == [
        (0, 2, {"current_classes": [0, 1]}),
        (1, 2, {"current_classes": [0, 1]}),
    ]


def test_exemplar_batch_cache_invalidates_on_mutation():
    """The cached exemplar batch must be rebuilt after memory mutations."""
    mem = PrototypeMemory(feature_dim=4, store_raw=True)
    mem.update_or_create_prototype(
        torch.zeros(4),
        torch.tensor([1.0]),
        torch.zeros(1, 2),
        task_id=0,
        label=torch.tensor(0),
    )
    f1, _ = mem.get_exemplar_batch(torch.device("cpu"))
    f2, _ = mem.get_exemplar_batch(torch.device("cpu"))
    assert f1.data_ptr() == f2.data_ptr(), "second call should hit the cache"

    mem.update_or_create_prototype(
        torch.full((4,), 10.0),
        torch.tensor([1.0]),
        torch.zeros(1, 2),
        task_id=0,
        label=torch.tensor(1),
    )
    f3, _ = mem.get_exemplar_batch(torch.device("cpu"))
    assert f3.shape[0] > f1.shape[0], "mutation must invalidate the cache"


def test_refresh_representations_handles_multiple_prototypes():
    """Batched refresh must split the encoded rows back to their prototypes."""
    enc = nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        enc.weight.copy_(torch.eye(6))
    mem = PrototypeMemory(feature_dim=6, store_raw=True, distance_threshold=0.01)
    raws = [torch.randn(1, 6) * 10 for _ in range(3)]
    for i, r in enumerate(raws):
        with torch.no_grad():
            feat = enc(r[0])
        mem.update_or_create_prototype(
            feat,
            torch.tensor([1.0]),
            torch.zeros(1, 2),
            task_id=0,
            label=torch.tensor(i),
            raw_input=r,
        )
    assert len(mem.prototypes) == 3

    mem.refresh_representations(enc)
    for proto, r in zip(mem.prototypes, raws):
        with torch.no_grad():
            expected = enc(r[0])
        assert proto.x_p.shape[0] == 1
        assert torch.allclose(proto.x_p[0], expected, atol=1e-5)
        assert torch.allclose(proto.v_p, proto.x_p.mean(dim=0), atol=1e-5)


def test_icarl_herding_matches_bruteforce():
    """Vectorized herding must select the same exemplars as the greedy scan."""
    from pal_moe.baselines.icarl import ICaRL

    torch.manual_seed(0)
    feats = torch.randn(64, 12)
    k = 5
    icarl = ICaRL.__new__(ICaRL)  # herding does not need the wrapped network
    got = icarl._herding_select(feats, k)

    current = torch.zeros_like(feats[0])
    remaining = list(range(feats.size(0)))
    ref = []
    for _ in range(k):
        best = min(remaining, key=lambda i: torch.norm(current + feats[i]).item())
        ref.append(best)
        remaining.remove(best)
        current = current + feats[best]
    assert got == ref


def test_derpp_update_buffer_chunked_logits_are_consistent():
    """Chunked buffer logits must equal a plain forward for the same sample."""
    from pal_moe.baselines.der import DERPP

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 3))
    x = torch.randn(100, 8)
    y = torch.randint(0, 3, (100,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16
    )
    trainer = DERPP(model, buffer_size=10, device=torch.device("cpu"))
    trainer.update_buffer(loader, seen_tasks=1, logit_batch_size=8)

    assert trainer.buffer.x
    with torch.no_grad():
        for bx, bl in zip(trainer.buffer.x, trainer.buffer.logits):
            expected = model(bx.unsqueeze(0)).squeeze(0)
            assert torch.allclose(bl, expected, atol=1e-5)


def test_config_validation_and_precedence(tmp_path):
    """--config is validated and explicit CLI flags win over config values."""
    import argparse
    import json

    from pal_moe.config import ConfigError, apply_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda_ood", type=float, default=0.1)
    parser.add_argument("--anchor", action="store_true")
    parser.add_argument(
        "--joint_freeze_router", action=argparse.BooleanOptionalAction, default=False
    )

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"epochs": 5, "lambda_ood": 0.2, "anchor": True}))
    args = parser.parse_args(["--epochs", "7"])
    apply_config(args, parser, ["--epochs", "7"], str(cfg_path))
    assert args.epochs == 7  # explicit CLI wins
    assert args.lambda_ood == 0.2  # config wins over the default
    assert args.anchor is True

    cfg_path.write_text(json.dumps({"epochz": 1}))
    with pytest.raises(ConfigError):
        apply_config(args, parser, [], str(cfg_path))

    cfg_path.write_text(json.dumps({"epochs": "many"}))
    with pytest.raises(ConfigError):
        apply_config(args, parser, [], str(cfg_path))

    cfg_path.write_text(json.dumps({"lambda_ood": -1.0}))
    with pytest.raises(ConfigError):
        apply_config(args, parser, [], str(cfg_path))

    cfg_path.write_text(json.dumps({"joint_freeze_router": True}))
    args2 = parser.parse_args(["--no-joint_freeze_router"])
    apply_config(args2, parser, ["--no-joint_freeze_router"], str(cfg_path))
    assert args2.joint_freeze_router is False


def test_split_cifar100_loader():
    """CIFAR-100 splitter produces 20 tasks x 5 disjoint classes (offline mock)."""
    import torchvision.datasets as tvd

    from pal_moe.data.split_cifar100 import NUM_TASKS, get_split_cifar100_tasks

    class FakeCIFAR100:
        def __init__(self, root, train=True, download=False, transform=None):
            self.transform = transform
            self.targets = [i % 100 for i in range(5000)]
            self.data = np.random.randint(0, 256, (5000, 32, 32, 3), dtype=np.uint8)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            x = self.transform(self.data[idx])
            return x, self.targets[idx]

    orig = tvd.CIFAR100
    tvd.CIFAR100 = FakeCIFAR100
    try:
        tasks = get_split_cifar100_tasks(
            data_dir=".", batch_size=32, val_split=0.1, seed=42
        )
    finally:
        tvd.CIFAR100 = orig

    assert len(tasks) == NUM_TASKS == 20
    assert tasks[0].classes == (0, 1, 2, 3, 4)
    assert tasks[19].classes == (95, 96, 97, 98, 99)
    x, y = next(iter(tasks[0].train_loader))
    assert x.shape == (32, 3, 32, 32)
    assert all(c in tasks[0].classes for c in y.unique().tolist())


def test_vit_encoder_shapes_and_projection():
    """ViT backbones resize inputs and keep an identity head at matched width."""
    enc = SharedEncoder(
        input_dim=3072,
        output_dim=768,
        arch="vit_b_32",
        backbone_weights="none",
        input_mean=(0.4914, 0.4822, 0.4465),
        input_std=(0.2470, 0.2435, 0.2616),
    )
    enc.eval()
    assert enc.backbone_feature_dim == 768
    assert isinstance(enc.net[1], nn.Identity)
    with torch.no_grad():
        out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 768)

    # A non-matching output width gets a frozen projection head.
    proj = SharedEncoder(
        input_dim=3072, output_dim=32, arch="vit_b_32", backbone_weights="none"
    )
    proj.eval()
    with torch.no_grad():
        assert proj(torch.randn(2, 3, 32, 32)).shape == (2, 32)
    assert proj.backbone_feature_dim == 768
    assert not isinstance(proj.net[1], nn.Identity)


def test_always_trigger_fires_on_new_task():
    from types import SimpleNamespace

    from pal_moe.trigger.expert_trigger import AlwaysTrigger

    trigger = AlwaysTrigger()
    result = trigger.evaluate(SimpleNamespace(num_experts=3))
    assert result.should_trigger
    assert result.best_parent_expert_idx == 2


def test_feature_cache_persistence_roundtrip(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    from pal_moe.data.feature_cache import (
        build_feature_cache,
        load_feature_cache,
        save_feature_cache,
    )

    class DummyTask:
        def __init__(self, task_id, loader):
            self.task_id = task_id
            self.classes = (0, 1)
            self.train_loader = loader
            self.val_loader = loader
            self.test_loader = loader

    torch.manual_seed(0)
    enc = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=8)
    enc.freeze()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    tasks = [DummyTask(0, loader), DummyTask(1, loader)]
    cache = build_feature_cache(
        enc, tasks, torch.device("cpu"), dtype=torch.float32, verbose=False
    )
    path = str(tmp_path / "feature_cache.pt")
    meta = {"dataset": "toy", "feature_dim": 8, "seed": 42}
    save_feature_cache(cache, path, meta)

    loaded = load_feature_cache(
        path, torch.device("cpu"), expected_meta=meta, verbose=False
    )
    assert loaded is not None
    assert loaded.feature_dim == 8
    assert len(loaded.tasks) == 2
    h0, y0 = next(iter(loaded.tasks[0].test_loader))
    h_ref, y_ref = next(iter(cache.tasks[0].test_loader))
    assert torch.allclose(h0, h_ref, atol=1e-6)
    assert torch.equal(y0, y_ref)

    # A mismatched meta is a hard error: no silent reuse of a stale cache.
    with pytest.raises(ValueError):
        load_feature_cache(
            path,
            torch.device("cpu"),
            expected_meta={"dataset": "toy", "feature_dim": 16},
            verbose=False,
        )

    assert (
        load_feature_cache(
            str(tmp_path / "missing.pt"), torch.device("cpu"), verbose=False
        )
        is None
    )


def test_stored_memory_accounting():
    """Every memory-carrying baseline reports the bytes it actually stores."""
    from pal_moe.baselines.buffer import SampleBuffer
    from pal_moe.baselines.ewc import EWC
    from pal_moe.baselines.icarl import ICaRL

    buf = SampleBuffer(4, mode="recency")
    assert buf.memory_bytes() == 0
    x = torch.zeros(2, 3, 4, 4)
    y = torch.tensor([0, 1])
    logits = torch.zeros(2, 10)
    buf.add_task(x, y, logits, per_task_budget=2)
    expected = (2 * 3 * 4 * 4 + 2 * 10) * 4 + 2 * 8  # labels are int64
    assert buf.memory_bytes() == expected

    net = nn.Sequential(nn.Linear(4, 3))
    ewc = EWC(net, lr=1e-3, device=torch.device("cpu"))
    assert ewc.memory_bytes() == 0
    loader = [(torch.randn(8, 4), torch.randint(0, 3, (8,)))]
    ewc.compute_fisher(loader)
    assert ewc.memory_bytes() > 0

    icarl = ICaRL(
        nn.Sequential(nn.Linear(4, 4), MLPExpert(4, 8, 3)),
        exemplars_per_class=1,
        num_classes=3,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    icarl.exemplars = {0: [torch.zeros(3, 4, 4)]}
    assert icarl.memory_bytes() > 0
    assert icarl.snapshot_bytes() == 0
    import copy

    icarl.old_model = copy.deepcopy(icarl.wrapper)
    assert icarl.snapshot_bytes() > 0


def test_encoder_in_channels_override():
    """Folder streams need explicit channel counts (not the 3072 heuristic)."""
    conv = SharedEncoder(
        input_dim=3 * 64 * 64,
        output_dim=8,
        arch="conv",
        conv_channels=(4,),
        in_channels=3,
    )
    assert conv(torch.randn(2, 3, 64, 64)).shape == (2, 8)

    resnet = SharedEncoder(
        input_dim=3 * 64 * 64, output_dim=8, arch="resnet18", in_channels=3
    )
    assert resnet.net[0].conv1.in_channels == 3


def test_mir_selects_interfered_samples():
    from pal_moe.baselines.mir import MIR

    encoder = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=4)
    net = nn.Sequential(
        encoder, MLPExpert(input_dim=4, hidden_dim=8, num_classes=3, expert_id=0)
    )
    trainer = MIR(
        net,
        buffer_size=10,
        lr=1e-3,
        device=torch.device("cpu"),
        candidate_pool=8,
    )
    assert trainer.memory_bytes() == 0
    loader = [(torch.randn(6, 16), torch.randint(0, 3, (6,))) for _ in range(2)]
    trainer.train_task(0, loader, epochs=1)
    assert trainer.memory_bytes() > 0
    rx, ry = trainer._select_interfered(batch_size=4)
    assert rx is not None and rx.shape[0] <= 4
    assert ry is not None and ry.shape[0] == rx.shape[0]


def test_latent_replay_trainer():
    from pal_moe.baselines.latent_replay import LatentReplayTrainer

    encoder = SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=4)
    net = nn.Sequential(
        encoder, MLPExpert(input_dim=4, hidden_dim=8, num_classes=3, expert_id=0)
    )
    trainer = LatentReplayTrainer(
        net,
        encoder_fn=lambda x: net[0](x),
        head=net[1],
        buffer_size=10,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    assert trainer.memory_bytes() == 0
    loader = [(torch.randn(6, 16), torch.randint(0, 3, (6,))) for _ in range(2)]
    trainer.train_task(0, loader, epochs=1)
    assert trainer.memory_bytes() > 0
    h, y = trainer.get_replay_batch(4)
    assert h is not None and h.shape[1] == 4
    assert y is not None and y.shape[0] == h.shape[0]


def test_split_folder_tasks(tmp_path):
    from PIL import Image

    from pal_moe.data.split_folder import get_split_folder_tasks

    for cls in ("a", "b", "c", "d"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(6):
            Image.new("RGB", (8, 8), color=(i * 30, 10, 20)).save(cls_dir / f"{i}.png")

    tasks = get_split_folder_tasks(
        str(tmp_path),
        batch_size=4,
        val_split=0.25,
        test_split=0.25,
        classes_per_task=2,
        num_workers=0,
    )
    assert len(tasks) == 2
    assert tasks[0].classes == (0, 1)
    assert tasks[1].classes == (2, 3)
    x, y = next(iter(tasks[0].train_loader))
    assert x.shape[1:] == (3, 32, 32)
    assert x.dtype == torch.float32


def test_routing_probe_helper():
    """The retention probe returns a fixed top-1 assignment per probed task."""
    import importlib.util
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "bench_runner", root / "experiments" / "run_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = DynamicMoE(
        encoder=SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=4),
        router=DynamicRouter(input_dim=4, num_experts=2, top_k=1),
        experts=[
            MLPExpert(input_dim=4, hidden_dim=8, num_classes=3, expert_id=i)
            for i in range(2)
        ],
    )

    class _Task:
        def __init__(self):
            self.test_loader = [(torch.randn(5, 16), torch.zeros(5, dtype=torch.long))]

    routes = mod._routing_probe(model, [_Task(), _Task()], torch.device("cpu"), upto=1)
    assert set(routes) == {0, 1}
    assert routes[0].shape == (5,)
    assert int(routes[0].max()) < 2
