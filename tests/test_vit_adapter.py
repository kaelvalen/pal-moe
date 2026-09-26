"""AdaptedViT: identity at init, frozen path untouched, one active expert per forward."""

import pytest
import torch

pytest.importorskip("timm")

from pal_moe.core.vit_adapter import AdaptedViT  # noqa: E402


def test_adapters_are_identity_at_init_and_frozen_path_is_untouched():
    torch.manual_seed(0)
    m = AdaptedViT("vit_tiny_patch16_224", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    m.add_expert("a")
    m.add_expert("b")
    with torch.no_grad():
        f0 = m(x)
        assert torch.equal(m(x, "a"), f0) and torch.equal(m(x, "b"), f0)
        torch.nn.init.normal_(m.adapters["a"][0].up.weight, std=0.1)
        fa = m(x, "a")
        assert not torch.equal(fa, f0)
        assert torch.equal(m(x), f0) and torch.equal(m(x, "b"), f0)
    assert all(not p.requires_grad for p in m.vit.parameters())
    m.freeze_expert("a")
    assert all(not p.requires_grad for p in m.adapters["a"].parameters())
    assert all(p.requires_grad for p in m.adapters["b"].parameters())
