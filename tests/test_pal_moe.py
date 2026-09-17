"""
Unit tests for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).
Verifies all mathematical formulations, loss computations, function-preserving expansion,
trigger thresholds, validation gate, and TTT.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pal_moe.adaptation.ttt import TestTimeAdapter
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
    assert (
        confidence.item() < 0.05
    ), f"ambiguous input should not be anchored: {confidence}"
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
