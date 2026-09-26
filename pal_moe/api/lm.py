"""`PalMoELM`: the v3 facade over a frozen HuggingFace causal LM (phase 3, no claims).

    FAST    `write("a sentence")`        key = hidden state at the key layer, value = the
                                         sentence; retrieved into the context at predict
                                         time when cosine >= `memory_threshold`.
    MEDIUM  `write(Batch(prompts, targets))`  closed-form edit of one MLP down-projection
                                         (`pal_moe.edit.DownProjEdit`), residuals against
                                         the base model, one float64 accumulator, forget =
                                         downdate.
    SLOW    `consolidate()`              frozen LoRA experts - declared, not built yet.

The four guards run exactly as on the vision path (`GuardedEditor`); canary outputs are
the next-token logits of a fixed prompt set through the full predict path (retrieval
included).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pal_moe.core.hashing import digest
from pal_moe.edit.down_proj import DownProjEdit
from pal_moe.memory.kv_store import FastMemory

from .guards import GuardConfig, GuardedEditor
from .records import Batch, EditRecord, Example


@dataclass
class LMPrediction:
    prompts: list[str]  # after retrieval (what the model actually saw)
    next_token: torch.Tensor  # [B]
    logits: torch.Tensor  # [B, V]
    retrieved: list[list[str]] = field(default_factory=list)
    texts: list[str] | None = None  # generations, when max_new_tokens > 0


class PalMoELM(GuardedEditor):
    def __init__(
        self,
        lm,
        prior,
        canary_prompts: list[str] | None = None,
        memory_threshold: float = 0.95,
        top_k: int = 1,
        guards: GuardConfig | None = None,
        value_steps: int = 20,
        value_lr: float = 0.5,
        edit_mode: str = "accumulate",
    ):
        """`prior` is a `KeyPrior` from `estimate_key_covariance(lm.collect_keys(corpus))`
        - corpus keys, never the edit keys (see `pal_moe.edit.down_proj`)."""
        super().__init__(lm.base_hash, guards)
        self.lm = lm
        self.memory = FastMemory(lm.hidden_size, device="cpu")
        self.edit = DownProjEdit(prior, mode=edit_mode)
        self.canary = list(canary_prompts or [])
        self.memory_threshold, self.top_k = float(memory_threshold), int(top_k)
        self.value_steps, self.value_lr = value_steps, value_lr

    # -- paths ---------------------------------------------------------------------

    def write(self, item, path: str | None = None) -> EditRecord:
        return self._guarded_write(item, path)

    def _install(self) -> None:
        delta = self.edit.solve()
        dtype = next(self.lm.model.parameters()).dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            dtype = torch.bfloat16  # quantised weights: compute dtype of the hook
        self.lm.set_delta(
            self.lm.edit_layer,
            None if delta is None else delta.to(self.lm.device, dtype),
        )

    def _apply(self, item, path):
        if isinstance(item, Batch) or path == "medium":
            b = item if isinstance(item, Batch) else Batch(*item)
            prompts, targets = list(b.x), list(b.y)
            with self.lm.base_only():
                K, Y0 = self.lm.down_proj_io(prompts)
                V = torch.stack(
                    [
                        self.lm.target_value(
                            p, t, steps=self.value_steps, lr=self.value_lr
                        )
                        for p, t in zip(prompts, targets)
                    ]
                )
            h = digest(prompts, targets, K, V)
            rid = self._next_id("medium", h)
            c = self.edit.contribution(rid, K, V - Y0)
            self.edit.add(c)
            self._install()
            return EditRecord(
                rid, "medium", h, meta={"prompts": prompts, "targets": targets}
            ), c
        text = item.x if isinstance(item, Example) else str(item)
        if isinstance(item, Example) and item.y is not None:
            text = f"{item.x} {item.y}"
        key = self.lm.keys([text])[0].cpu()
        h = digest(text, key)
        rid = self._next_id("fast", h)
        self.memory.write(rid, key, text)
        return EditRecord(rid, "fast", h, meta={"text": text}), None

    def _undo(self, record: EditRecord):
        if record.kind == "fast":
            token = self.memory.entry(record.id)
            self.memory.delete(record.id)
            return token
        if record.kind == "medium":
            c = self.edit.remove(record.id)
            self._install()
            return c
        raise KeyError(record.kind)

    def _redo(self, record: EditRecord, token) -> None:
        if record.kind == "fast":
            self.memory.write(record.id, *token)
        else:
            self.edit.add(token)
            self._install()

    def consolidate(self, policy: str = "by_arrival", prereg: str | None = None):
        raise NotImplementedError(
            "the LM slow path (frozen LoRA experts over consolidated edits) is declared in "
            "docs/V3_ARCHITECTURE.md but not built; phase 3 covers the fast and medium paths"
        )

    # -- read path -------------------------------------------------------------------

    def retrieve(self, prompts: list[str]) -> list[list[str]]:
        if not len(self.memory):
            return [[] for _ in prompts]
        keys = self.lm.keys(prompts).cpu()
        return [
            [h.value for h in hits]
            for hits in self.memory.lookup(
                keys, k=self.top_k, threshold=self.memory_threshold
            )
        ]

    @torch.no_grad()
    def predict(self, prompts: list[str], max_new_tokens: int = 0) -> LMPrediction:
        retrieved = self.retrieve(prompts)
        seen = [
            ("\n".join(r) + "\n" + p) if r else p for r, p in zip(retrieved, prompts)
        ]
        logits = self.lm.next_token_logits(seen)
        texts = self.lm.generate(seen, max_new_tokens) if max_new_tokens > 0 else None
        return LMPrediction(seen, logits.argmax(-1), logits, retrieved, texts)

    # -- guard hooks ----------------------------------------------------------------

    def _routers(self) -> list:
        return [self.memory.index]

    def _canary_outputs(self):
        if not self.canary:
            empty = torch.zeros(0)
            return empty, empty.long()
        p = self.predict(self.canary)
        return p.logits.double().cpu(), p.next_token.cpu()

    def _solution(self):
        return self.edit.solve()

    def _recompute_report(self) -> dict:
        rep = self.edit.recompute_report(tol=self.guards.tolerance)
        rep["hook_removed"] = not self.lm._deltas if not self.edit._arrival else None
        return rep

    def _order_report(self):
        ids = list(self.edit._arrival)
        if not ids:
            return digest(ids), {}
        rep = self.edit.order_report(tol=self.guards.tolerance)
        if self.canary:
            # Install the permuted-order solution, read the canary argmax, restore.
            ref = self._canary_outputs()[1]
            g = torch.Generator().manual_seed(len(ids))
            perm = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
            permuted = self.edit._solve_rows(perm)
            dtype = self.lm._deltas[self.lm.edit_layer % len(self.lm.layers)].dtype
            self.lm.set_delta(self.lm.edit_layer, permuted.to(self.lm.device, dtype))
            rep["argmax_identical"] = bool(torch.equal(ref, self._canary_outputs()[1]))
            rep["pass"] = rep["pass"] and rep["argmax_identical"]
            self._install()
        return digest(ids), rep
