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
from pal_moe.edit.down_proj import DownProjEdit, estimate_key_covariance  # noqa: E402


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


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "all that glitters is not gold",
    "where there is smoke there is fire",
    "the early bird catches the worm",
]


def _prior(lm, weight=10.0):
    keys = lm.collect_keys(CORPUS)  # every token of a corpus, never the edit keys
    assert keys.size(0) > 64
    return estimate_key_covariance(
        keys, weight=weight, ridge=1e-2, source="test corpus"
    )


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


def test_prior_must_come_from_enough_corpus_tokens(lm):
    edit_keys, _ = lm.down_proj_io(["the sky is"])
    with pytest.raises(ValueError):
        DownProjEdit(estimate_key_covariance(edit_keys, source="edit keys only"))
    with pytest.raises(TypeError):
        DownProjEdit(torch.eye(64, dtype=torch.float64))


def test_corpus_prior_protects_unseen_keys_better_than_an_edit_only_prior(lm):
    """MEMIT's locality mechanism on a tiny random model (a geometry check, not a
    claim): a corpus covariance prior steers the edit away from the directions text
    occupies; an edit-only prior moves along the edit key. Measured as RMS output
    change per unit of edit gain, on the corpus keys and on held-out text."""
    K_e, Y0 = lm.down_proj_io(["the sky is"])
    R = torch.randn_like(Y0, generator=torch.Generator().manual_seed(0))
    K0 = lm.collect_keys(CORPUS).double()
    held = lm.collect_keys(
        ["an apple a day keeps the doctor away", "better late than never"]
    ).double()
    priors = (
        estimate_key_covariance(K0, weight=1e3, ridge=1e-2),
        estimate_key_covariance(K_e.repeat(80, 1), weight=1e3, ridge=1e-2),
    )
    rms = []
    for pr in priors:
        ed = DownProjEdit(pr)
        ed.add(ed.contribution("e", K_e, R))
        D = ed.solve()
        gain = float((K_e.double() @ D).norm())
        rms.append(
            [float((K @ D).pow(2).sum(1).mean().sqrt()) / gain for K in (K0, held)]
        )
    assert rms[0][0] < 0.7 * rms[1][0] and rms[0][1] < 0.7 * rms[1][1], rms


def test_lm_forget_is_downdate_and_forget_all_removes_the_hook(lm):
    m = PalMoELM(
        lm,
        _prior(lm),
        canary_prompts=["zzz qqq", "xyz"],
        value_steps=5,
        guards=GuardConfig(tolerance=1e-8),
    )
    base = lm.next_token_logits(["zzz qqq"])
    r1 = m.write(Batch(["the sky is"], ["g"]))
    r2 = m.write(Batch(["grass is"], ["b"]))
    rep = m.forget(r1.id)
    assert rep["pass"] and rep["downdate"]["max_abs_dDelta_recompute"] <= 1e-8
    m.forget(r2.id)
    assert not lm._deltas and torch.equal(lm.next_token_logits(["zzz qqq"]), base)
