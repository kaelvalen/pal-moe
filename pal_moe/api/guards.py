"""The four guards and the edit-log machinery shared by every v3 backend.

Run on every `write` / `forget` / `consolidate`:

1. **Locality** - output change on a fixed canary set <= epsilon (per path, logged).
2. **Reversibility** - `write(x); forget(id)` restores `state()`, the weights and all
   canary outputs **bitwise**. Checked two ways: a trial undo/redo inside every
   `write`, and on every `forget` against the digests recorded the last time the
   system was in the resulting state.
3. **Order invariance** - the medium path's canonical solution vs the arrival-order
   running sum: argmax identical on the canary set, max|dW| reported.
4. **Router purity** - router trainable parameter count == 0, asserted.

A backend subclasses `GuardedEditor` and implements the `_apply_* / _undo / _redo`
hooks; this class owns the log, the state hash and the guard bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pal_moe.core.hashing import digest
from pal_moe.router.base import assert_pure

from .records import EditRecord, GuardViolation, ReversibilityError, StateHash


@dataclass
class GuardConfig:
    epsilon_fast: float | None = 0.0  # max canary argmax flip rate for a FAST write
    epsilon_medium: float | None = None  # None: measure and log only
    epsilon_consolidation: float | None = None
    trial_reversibility: bool = True
    on_violation: str = "raise"  # "raise" (after rolling back) | "log"


class GuardedEditor:
    """Edit log, state hash and guards; backend-agnostic."""

    def __init__(self, base_hash: str, guards: GuardConfig | None = None):
        self.base_hash = base_hash
        self.guards = guards or GuardConfig()
        self.log: list[EditRecord] = []
        self._seq = 0
        self._seen: dict[
            str, tuple[str, str]
        ] = {}  # state digest -> (weights, outputs)
        self.guard_log: list[dict] = []

    # -- hooks a backend implements ---------------------------------------

    def _apply(self, item, path: str | None) -> tuple[EditRecord, object]:
        raise NotImplementedError

    def _undo(self, record: EditRecord) -> object:
        """Remove an applied edit; return a token that `_redo` can re-apply."""
        raise NotImplementedError

    def _redo(self, record: EditRecord, token: object) -> None:
        raise NotImplementedError

    def _canary_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits or scores [N, ...], labels [N]) on the fixed canary set."""
        raise NotImplementedError

    def _weights_digest(self) -> str:
        raise NotImplementedError

    def _order_report(self) -> tuple[str, dict]:
        return "", {}

    def _routers(self) -> list:
        return []

    # -- state ------------------------------------------------------------

    def state(self) -> StateHash:
        edits = tuple((r.id, r.kind, r.content_hash) for r in self.log)
        return StateHash(self.base_hash, edits, digest(self.base_hash, list(edits)))

    def _next_id(self, kind: str, content_hash: str) -> str:
        self._seq += 1
        return f"{kind}-{self._seq:06d}-{content_hash[:12]}"

    def _outputs_digest(self) -> tuple[str, torch.Tensor, torch.Tensor]:
        scores, labels = self._canary_outputs()
        return digest(scores, labels), scores, labels

    def _remember_state(self) -> None:
        self._seen[self.state().digest] = (
            self._weights_digest(),
            self._outputs_digest()[0],
        )

    # -- guards -----------------------------------------------------------

    def check_purity(self) -> dict:
        counts = {
            getattr(r, "name", type(r).__name__): assert_pure(r)
            for r in self._routers()
        }
        return {"router_trainable_params": counts, "pass": True}

    @staticmethod
    def locality(before, after, epsilon: float | None) -> dict:
        (s0, l0), (s1, l1) = before, after
        n = max(1, l0.numel())
        flips = int((l0 != l1).sum())
        rep = {
            "canary_size": int(l0.numel()),
            "argmax_flips": flips,
            "flip_rate": flips / n,
            "max_abs_output_delta": float((s1.double() - s0.double()).abs().max())
            if s0.numel()
            else 0.0,
            "epsilon": epsilon,
        }
        rep["pass"] = epsilon is None or rep["flip_rate"] <= epsilon
        return rep

    def _violate(
        self, guard: str, report: dict, record: EditRecord | None, rollback: bool
    ) -> None:
        self.guard_log.append(
            {"guard": guard, "edit": record.id if record else None, **report}
        )
        if self.guards.on_violation != "raise":
            return
        if rollback and record is not None and record in self.log:
            self._undo(record)
            self.log.remove(record)
        err = ReversibilityError if guard == "reversibility" else GuardViolation
        raise err(guard, report)

    def _epsilon(self, kind: str) -> float | None:
        return {
            "fast": self.guards.epsilon_fast,
            "medium": self.guards.epsilon_medium,
            "consolidation": self.guards.epsilon_consolidation,
        }[kind]

    def _trial_reversibility(
        self, record: EditRecord, pre_w: str, pre_o: str, post_o: str
    ) -> dict:
        token = self._undo(record)
        self.log.remove(record)
        undo_w, undo_o = self._weights_digest(), self._outputs_digest()[0]
        self._redo(record, token)
        self.log.append(record)
        redo_o = self._outputs_digest()[0]
        rep = {
            "undo_restores_weights": undo_w == pre_w,
            "undo_restores_canary": undo_o == pre_o,
            "redo_restores_canary": redo_o == post_o,
        }
        rep["pass"] = all(rep.values())
        return rep

    # -- public operations --------------------------------------------------

    def _guarded_write(self, item, path: str | None = None) -> EditRecord:
        purity = self.check_purity()
        if not self._seen:
            self._remember_state()
        pre_w = self._weights_digest()
        pre_o, s0, l0 = self._outputs_digest()
        record, _ = self._apply(item, path)
        self.log.append(record)
        record.purity_report = purity
        post_o, s1, l1 = self._outputs_digest()
        record.locality_report = self.locality(
            (s0, l0), (s1, l1), self._epsilon(record.kind)
        )
        record.order_hash, record.order_report = self._order_report()
        if self.guards.trial_reversibility:
            record.reversibility_report = self._trial_reversibility(
                record, pre_w, pre_o, post_o
            )
        self.check_purity()
        if not record.locality_report["pass"]:
            self._violate("locality", record.locality_report, record, rollback=True)
        if record.reversibility_report and not record.reversibility_report["pass"]:
            self._violate(
                "reversibility", record.reversibility_report, record, rollback=True
            )
        if record.order_report and not record.order_report.get(
            "argmax_identical", True
        ):
            self._violate(
                "order_invariance", record.order_report, record, rollback=True
            )
        self._remember_state()
        return record

    def forget(self, edit_id: str) -> dict:
        """Exact removal: memory delete / statistics downdate / expert drop."""
        self.check_purity()
        record = next((r for r in self.log if r.id == edit_id), None)
        if record is None:
            raise KeyError(f"no live edit {edit_id!r}")
        self._undo(record)
        self.log.remove(record)
        self.check_purity()
        st = self.state()
        w, o = self._weights_digest(), self._outputs_digest()[0]
        rep = {"edit": edit_id, "state_seen_before": st.digest in self._seen}
        if rep["state_seen_before"]:
            ref_w, ref_o = self._seen[st.digest]
            rep["weights_bitwise"] = w == ref_w
            rep["canary_bitwise"] = o == ref_o
            rep["pass"] = rep["weights_bitwise"] and rep["canary_bitwise"]
        else:
            rep["pass"] = True  # a state never visited has no reference to compare to
            self._seen[st.digest] = (w, o)
        if not rep["pass"]:
            self._violate("reversibility", rep, None, rollback=False)
        return rep
