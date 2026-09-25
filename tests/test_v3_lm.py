"""v3 LM backend (phase 3): hooks, fast path, closed-form medium edit, the guards.

A tiny randomly initialised Llama and a character tokenizer - infrastructure checks
only, no claim about editing quality.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pal_moe.api import Batch, GuardConfig  # noqa: E402
from pal_moe.api.lm import PalMoELM  # noqa: E402
from pal_moe.core.hf_lm import HFCausalLM  # noqa: E402
from pal_moe.edit.down_proj import estimate_key_covariance  # noqa: E402


class CharTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def _ids(self, text):
        return [2 + (ord(ch) % 60) for ch in text]

    def __call__(
        self, texts, return_tensors="pt", padding=False, add_special_tokens=True
    ):
        texts = [texts] if isinstance(texts, str) else list(texts)
        ids = [self._ids(t) for t in texts]
        n = max(len(i) for i in ids)
        return {
            "input_ids": torch.tensor([[0] * (n - len(i)) + i for i in ids]),
            "attention_mask": torch.tensor(
                [[0] * (n - len(i)) + [1] * len(i) for i in ids]
            ),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(ord("a") + (int(i) - 2) % 26) for i in ids if int(i) > 1)


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = transformers.LlamaForCausalLM(cfg).float()
    return HFCausalLM(
        model, CharTokenizer(), key_layers=(-1,), edit_layer=-1, hash_weights=True
    )


def _prior(lm):
    corpus = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "a stitch in time",
        "saves nine",
    ]
    keys = torch.cat(
        [lm.down_proj_io([t[:j]])[0] for t in corpus for j in range(3, len(t))]
    )
    return estimate_key_covariance(keys, weight=10.0, ridge=1e-2)


def test_hooks_expose_keys_and_down_proj_io(lm):
    k = lm.keys(["hello world", "hi"])
    x, y = lm.down_proj_io(["hello world"])
    assert k.shape == (2, 32) and x.shape == (1, 64) and y.shape == (1, 32)
    before = lm.next_token_logits(["hello"])
    lm.set_delta(-1, torch.randn(64, 32) * 0.1)
    assert not torch.equal(lm.next_token_logits(["hello"]), before)
    lm.set_delta(-1, None)
    assert torch.equal(lm.next_token_logits(["hello"]), before)


def test_lm_fast_write_retrieve_forget_is_bitwise(lm):
    m = PalMoELM(
        lm,
        _prior(lm),
        canary_prompts=["zzz qqq", "xyz"],
        memory_threshold=0.999,
        guards=GuardConfig(epsilon_fast=0.0),
    )
    s0, p0 = m.state(), m.predict(["the capital of france is"])
    rec = m.write("the capital of france is paris")
    assert rec.locality_report["pass"] and rec.reversibility_report["pass"]
    m.forget(rec.id)
    p1 = m.predict(["the capital of france is"])
    assert m.state() == s0 and torch.equal(p0.logits, p1.logits)


def test_lm_medium_edit_changes_target_and_forgets_exactly(lm):
    m = PalMoELM(
        lm,
        _prior(lm),
        canary_prompts=["zzz qqq", "xyz"],
        value_steps=30,
        guards=GuardConfig(epsilon_medium=None),
    )
    prompt, target = "the sky is", "g"
    before = lm.target_logprob([prompt], [target])
    d0 = m._outputs_digest()[0]
    rec = m.write(Batch([prompt], [target]))
    after = lm.target_logprob([prompt], [target])
    assert (
        after > before
    )  # the edit moves the target (infrastructure check, not a claim)
    assert rec.reversibility_report["pass"] and rec.purity_report["pass"]
    rec2 = m.write(Batch(["grass is"], ["b"]))
    assert rec2.order_report["argmax_identical"]
    m.forget(rec2.id)
    m.forget(rec.id)
    assert m._outputs_digest()[0] == d0 and not lm._deltas


def test_lm_medium_is_order_invariant(lm):
    pairs = [("the sky is", "g"), ("grass is", "b"), ("snow is", "k")]
    digests = []
    for order in (pairs, list(reversed(pairs))):
        m = PalMoELM(
            lm, _prior(lm), value_steps=5, guards=GuardConfig(trial_reversibility=False)
        )
        recs = [m.write(Batch([p], [t])) for p, t in order]
        digests.append((m.edit.solve().clone(), lm.next_token_logits(["the sky is"])))
        for r in recs:
            m.forget(r.id)
    assert torch.equal(digests[0][0], digests[1][0]) and torch.equal(
        digests[0][1], digests[1][1]
    )
