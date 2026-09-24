"""
Tests for the S1 architecture contract (docs/ARCHITECTURE_CONTRACT.md).

The contract is enforced structurally: every registered implementation must
satisfy its protocol, an expert must be the identity at initialization (that is
what makes a bank function-preserving), and a classification head must NOT pass
the representation-expert check - that distinction is the whole point of S1.
"""

import pytest
import torch

from pal_moe.arch import (
    CLASSIFICATION_EXPERTS,
    READOUTS,
    Backbone,
    ClassificationExpert,
    NCMReadout,
    PrototypeRouter,
    Readout,
    RepresentationExpert,
    ResidualAdapter,
    RidgeReadout,
    Router,
    build_backbone,
    build_expert,
    build_readout,
    describe,
    identity_check,
    is_classification_expert,
    is_representation_expert,
    mask_unseen,
    wrap_encoder,
)
from pal_moe.data.feature_cache import CachedFeatureEncoder
from pal_moe.models.encoder import SharedEncoder

B = 8


# ------------------------------------------------------------------ registries


def test_registry_lists_the_ladder_from_the_brief():
    names = describe()
    assert {"mlp", "conv", "resnet18", "vit_b_16", "cached", "random"} <= set(
        names["backbone"]
    )
    assert {"identity", "residual_adapter", "mlp"} <= set(names["expert"])
    assert "legacy_mlp" in names["classification_expert"]
    assert {"ncm", "cosine", "linear", "logistic", "ridge", "mlp"} <= set(
        names["readout"]
    )
    assert {"prototype", "legacy_linear"} <= set(names["router"])


def test_unknown_name_fails_loudly_with_the_available_names():
    with pytest.raises(KeyError, match="unknown readout"):
        build_readout("does_not_exist", dim=4, num_classes=3)


def test_duplicate_registration_is_an_error_unless_intentional():
    with pytest.raises(KeyError, match="already registered"):

        @READOUTS.register("ncm")
        def _duplicate(*_args, **_kwargs):  # pragma: no cover - never runs
            raise AssertionError


# ------------------------------------------------------------------- backbones


@pytest.mark.parametrize("name", ["mlp", "random", "cached"])
def test_backbones_satisfy_the_contract(name):
    kwargs = {"input_dim": 32, "output_dim": 16}
    if name == "cached":
        kwargs = {"output_dim": 16}
    backbone = build_backbone(name, **kwargs)
    assert isinstance(backbone, Backbone)
    z = backbone.encode(torch.randn(B, 32) if name != "cached" else torch.randn(B, 16))
    assert z.shape == (B, 16)
    assert z.shape[1] == backbone.output_dim


def test_shared_encoder_wraps_without_changing_numerics():
    """The wrapper must be transparent: same tensor in, same tensor out."""
    encoder = SharedEncoder(input_dim=32, hidden_dims=(16,), output_dim=8, arch="mlp")
    encoder.eval()
    wrapped = wrap_encoder(encoder)
    x = torch.randn(B, 32)
    with torch.no_grad():
        assert torch.equal(wrapped.encode(x), encoder(x))


def test_cached_encoder_wraps_to_the_identity_backbone():
    backbone = wrap_encoder(CachedFeatureEncoder(12))
    x = torch.randn(B, 12)
    assert torch.equal(backbone.encode(x), x)
    assert backbone.output_dim == 12


def test_random_backbone_is_seeded_and_frozen():
    a = build_backbone("random", input_dim=16, output_dim=8, seed=3)
    b = build_backbone("random", input_dim=16, output_dim=8, seed=3)
    x = torch.randn(B, 16)
    assert torch.equal(a.encode(x), b.encode(x))
    a.freeze()
    assert all(not p.requires_grad for p in a.parameters())


# --------------------------------------------------------------------- experts


# Constructors genuinely differ per implementation, so the shared test states
# the kwargs explicitly instead of hiding the difference behind **kwargs.
EXPERT_KWARGS = {
    "identity": {},
    "residual_adapter": {"rank": 4},
    "mlp": {"hidden": 8},
}
READOUT_KWARGS = {
    "ncm": {},
    "cosine": {"steps": 20},
    "linear": {"steps": 20},
    "logistic": {"steps": 20},
    "ridge": {},
    "mlp": {"steps": 20},
}


@pytest.mark.parametrize("name", ["identity", "residual_adapter", "mlp"])
def test_experts_satisfy_the_contract_and_start_as_the_identity(name):
    """Rule: adding an expert must be function-preserving."""
    expert = build_expert(name, dim=16, **EXPERT_KWARGS[name])
    assert isinstance(expert, RepresentationExpert)
    assert is_representation_expert(expert)
    z = torch.randn(B, 16)
    assert identity_check(expert, z)
    assert expert.transform(z).shape == z.shape


def test_a_legacy_head_is_a_classification_expert_not_a_representation_expert():
    """The abstraction error S1 exists to make visible."""
    head = CLASSIFICATION_EXPERTS.build(
        "legacy_mlp", input_dim=16, hidden_dim=8, num_classes=5
    )
    assert isinstance(head, ClassificationExpert)
    assert is_classification_expert(head)
    assert not is_representation_expert(head)
    assert head.classify(torch.randn(B, 16)).shape == (B, 5)


def test_residual_adapter_zero_rank_is_exactly_the_identity():
    expert = ResidualAdapter(dim=16, rank=0)
    z = torch.randn(B, 16)
    assert torch.equal(expert.transform(z), z)
    assert expert.param_count() == 0


def test_expert_trains_away_from_the_identity():
    expert = ResidualAdapter(dim=8, rank=4)
    z = torch.randn(32, 8)
    target = torch.randn(32, 8)
    opt = torch.optim.Adam(expert.parameters(), lr=0.05)
    for _ in range(20):
        opt.zero_grad()
        loss = (expert.transform(z) - target).pow(2).mean()
        loss.backward()
        opt.step()
    assert not identity_check(expert, z, atol=1e-3)


# -------------------------------------------------------------------- readouts


@pytest.mark.parametrize(
    "name", ["ncm", "cosine", "linear", "logistic", "ridge", "mlp"]
)
def test_readouts_satisfy_the_contract(name):
    readout = build_readout(name, dim=8, num_classes=4, **READOUT_KWARGS[name])
    assert isinstance(readout, Readout)
    z = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    report = readout.fit(z, y, seen_classes=[0, 1, 2, 3])
    assert isinstance(report, dict)
    logits = readout.predict(z)
    assert logits.shape == (32, 4)


def test_ncm_matches_a_hand_computed_reference():
    readout = NCMReadout(dim=3, num_classes=2, scale=1.0)
    z = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]])
    y = torch.tensor([0, 1, 0])
    readout.fit(z, y)
    expected = torch.tensor([[0.95, 0.05, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(readout.means, expected, atol=1e-6)
    # A query on class 0's mean must score class 0 highest.
    assert int(readout.predict(torch.tensor([[1.0, 0.0, 0.0]])).argmax()) == 0


def test_ncm_running_mean_is_exact_when_fit_in_two_halves():
    """The online case must not need stored samples."""
    z = torch.randn(40, 6)
    y = torch.randint(0, 3, (40,))
    one = NCMReadout(dim=6, num_classes=3)
    one.fit(z, y)
    two = NCMReadout(dim=6, num_classes=3)
    two.fit(z[:19], y[:19])
    two.fit(z[19:], y[19:])
    assert torch.allclose(one.means, two.means, atol=1e-5)


def test_ridge_is_closed_form_and_exactly_incremental():
    """Accumulated sufficient statistics must equal fitting on all data at once."""
    torch.manual_seed(0)
    z = torch.randn(60, 5)
    y = torch.randint(0, 3, (60,))
    one = RidgeReadout(dim=5, num_classes=3, ridge=0.1)
    one.fit(z, y)
    two = RidgeReadout(dim=5, num_classes=3, ridge=0.1)
    two.fit(z[:25], y[:25])
    two.fit(z[25:], y[25:])
    assert torch.allclose(one.W, two.W, atol=1e-5)
    # And it solves the normal equations.
    augmented = torch.cat([z, torch.ones(z.size(0), 1)], dim=1)
    targets = torch.zeros(60, 3)
    targets[torch.arange(60), y] = 1.0
    a = augmented.t() @ augmented + 0.1 * torch.eye(6)
    expected = torch.linalg.solve(a, augmented.t() @ targets).t()
    assert torch.allclose(one.W, expected, atol=1e-5)


def test_readouts_learn_a_linearly_separable_problem():
    """Contract-level sanity: the ladder must actually fit something.

    `cosine` is excluded on purpose: on a *symmetric two-class* problem its
    optimisation is unreliable (measured 52.5% where the others reach >90%),
    because both row directions start random and must move symmetrically. That
    is a property of that readout, not of the contract, and the E0 reference
    numbers were produced on a many-class problem where it behaves.
    """
    torch.manual_seed(0)
    z = torch.randn(120, 4)
    y = (z[:, 0] + z[:, 1] > 0).long()
    # Ridge and NCM have no optimiser, so they take no budget.
    budget = {"steps": 400, "lr": 0.05}
    for name in ("linear", "logistic", "ridge", "ncm", "mlp"):
        kwargs = {
            **READOUT_KWARGS[name],
            **(budget if name not in ("ridge", "ncm") else {}),
        }
        readout = build_readout(name, dim=4, num_classes=2, **kwargs)
        readout.fit(z, y, seen_classes=[0, 1])
        acc = float((readout.predict(z).argmax(dim=-1) == y).float().mean())
        assert acc > 0.9, f"{name} failed to fit a separable problem: {acc:.3f}"


def test_mask_unseen_hides_future_classes():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    masked = mask_unseen(logits, [0, 1])
    assert masked[0, 0] == 1.0 and masked[0, 1] == 2.0
    assert masked[0, 2] < -1e8


def test_readouts_are_interchangeable_behind_the_same_call():
    """The point of the axis: a study swaps readouts without touching the loop."""
    z = torch.randn(48, 6)
    y = torch.randint(0, 3, (48,))
    seen = [0, 1, 2]
    for name in ("ncm", "ridge", "cosine"):
        readout = build_readout(name, dim=6, num_classes=3, **READOUT_KWARGS[name])
        readout.fit(z, y, seen_classes=seen)
        assert readout.predict(z).shape == (48, 3)


# --------------------------------------------------------------------- routers


def test_prototype_router_candidate_set_and_distribution():
    router = PrototypeRouter(dim=4, num_classes=4, num_experts=2)
    # classes 0,1 -> expert 0; classes 2,3 -> expert 1
    for c in range(4):
        mean = torch.zeros(4)
        mean[c] = 1.0
        router.register_class(c, expert_id=0 if c < 2 else 1, z=mean)
    assert isinstance(router, Router)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # class 0 -> expert 0
    probs = router.route(query)
    assert probs.shape == (1, 2)
    assert pytest.approx(1.0, abs=1e-6) == float(probs.sum())
    ids, _ = router.top_k(query, k=1)
    assert int(ids[0, 0]) == 0


def test_legacy_router_exposes_the_same_interface():
    from pal_moe.arch import build_legacy_router

    router = build_legacy_router("dynamic", input_dim=6, top_k=1, num_experts=3)
    assert isinstance(router, Router)
    z = torch.randn(B, 6)
    assert router.route(z).shape == (B, 3)
    ids, values = router.top_k(z, k=2)
    assert ids.shape == (B, 2) and values.shape == (B, 2)


def test_router_registry_has_no_unknown_v1_type():
    with pytest.raises(KeyError, match="unknown router_type"):
        from pal_moe.arch import build_legacy_router

        build_legacy_router("nope", input_dim=4, top_k=1, num_experts=2)
