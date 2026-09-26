"""The four guards and the edit-log machinery shared by every v3 backend.

Run on every `write` / `forget` / `consolidate`, and each one is a **measurement**:

1. **Locality** - canary argmax flip rate <= epsilon (per path, logged), plus the
   max |delta output|.
2. **Reversibility** - inside every write, a trial undo/redo: after the undo the
   medium-path solution must be within `tolerance` of the pre-write solution and the
   canary argmax identical (max |delta output| reported); on every forget, the
   accumulator must match the live contributions re-summed from scratch (the downdate
   drift) and, when the resulting state was visited before, the canary outputs must
   match that visit within `output_tolerance`. `bitwise` is reported separately and is
   only expected where it is real: the FAST path (a row is physically removed), a
   consolidation (the bank object is swapped back), and a medium path returned to
   zero edits (accumulators reset to the prior, the LM hook removed).
3. **Order invariance** - the accumulator vs the live contributions re-summed in a
   seeded random permutation: max|dW| <= `tolerance`, canary argmax identical.
4. **Router purity** - trainable scalars reachable from the router == 0 (inspects the
   object, not its self-report).

A backend subclasses `GuardedEditor` and implements the `_apply / _undo / _redo`
hooks plus `_solution`, `_recompute_report`, `_order_report`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pal_moe.core.hashing import digest
from pal_moe.router.base import assert_pure

from .records import EditRecord, GuardViolation, ReversibilityError, StateHash

_FINGERPRINT_MAX = (
    2_000_000  # canary scores kept per visited state above this: row maxima
)


@dataclass
class GuardConfig:
    epsilon_fast: float | None = 0.0  # max canary argmax flip rate for a FAST write
    epsilon_medium: float | None = None  # None: measure and log only
    epsilon_consolidation: float | None = None
    tolerance: float = (
        1e-10  # max |dW| for order invariance and reversibility (float64)
    )
    output_tolerance: float = 1e-6  # max |delta canary output| for reversibility
    trial_reversibility: bool = True
    order_check: bool = True
    on_violation: str = "raise"  # "raise" (after rolling back) | "log"


def _max_abs(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if a.shape != b.shape:
        return float("inf")
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


class GuardedEditor:
    """Edit log, state hash and guards; backend-agnostic."""

    def __init__(self, base_hash: str, guards: GuardConfig | None = None):
        self.base_hash = base_hash
        self.guards = guards or GuardConfig()
        self.log: list[EditRecord] = []
        self._seq = 0
        self._seen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}  # state -> canary
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
        """(scores [N, ...], labels [N]) on the fixed canary set."""
        raise NotImplementedError

    def _solution(self) -> torch.Tensor | None:
        """The medium path's current solved weights (or None when there are none)."""
        return None

    def _recompute_report(self) -> dict:
        return {"pass": True}

    def _order_report(self) -> tuple[str, dict]:
        return "", {}

    def _routers(self) -> list:
        return []

    def _weights_digest(self) -> str:
        return ""

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

    @staticmethod
    def _fingerprint(scores: torch.Tensor, labels: torch.Tensor):
        if scores.numel() > _FINGERPRINT_MAX:
            scores = scores.max(-1).values
        return scores.detach().double().cpu().clone(), labels.detach().cpu().clone()

    def _remember_state(self, scores=None, labels=None) -> None:
        if scores is None:
            _, scores, labels = self._outputs_digest()
        self._seen[self.state().digest] = self._fingerprint(scores, labels)

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
            "max_abs_output_delta": _max_abs(s0, s1),
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

    def _compare_outputs(self, ref, cur) -> dict:
        (s0, l0), (s1, l1) = ref, cur
        d = _max_abs(s0, s1)
        return {
            "canary_argmax_identical": bool(torch.equal(l0, l1)),
            "canary_max_abs_delta": d,
            "canary_bitwise": d == 0.0 and bool(torch.equal(l0, l1)),
        }

    def _trial_reversibility(
        self, record: EditRecord, pre_sol, pre_out, post_out
    ) -> dict:
        token = self._undo(record)
        self.log.remove(record)
        undo_sol = self._solution()
        undo_out = self._canary_outputs()
        rep = {"undo_max_abs_dW": _max_abs(pre_sol, undo_sol)}
        rep.update(
            {
                f"undo_{k}": v
                for k, v in self._compare_outputs(pre_out, undo_out).items()
            }
        )
        self._redo(record, token)
        self.log.append(record)
        rep.update(
            {
                f"redo_{k}": v
                for k, v in self._compare_outputs(
                    post_out, self._canary_outputs()
                ).items()
            }
        )
        rep["bitwise"] = rep["undo_max_abs_dW"] == 0.0 and rep["undo_canary_bitwise"]
        rep["pass"] = (
            rep["undo_max_abs_dW"] <= self.guards.tolerance
            and rep["undo_canary_argmax_identical"]
            and rep["undo_canary_max_abs_delta"] <= self.guards.output_tolerance
            and rep["redo_canary_argmax_identical"]
        )
        return rep

    # -- public operations --------------------------------------------------

    def _guarded_write(self, item, path: str | None = None) -> EditRecord:
        purity = self.check_purity()
        pre_out = self._canary_outputs()
        if not self._seen:
            self._remember_state(*pre_out)
        pre_sol = self._solution()
        pre_sol = None if pre_sol is None else pre_sol.clone()
        record, _ = self._apply(item, path)
        self.log.append(record)
        record.purity_report = purity
        post_out = self._canary_outputs()
        record.locality_report = self.locality(
            pre_out, post_out, self._epsilon(record.kind)
        )
        if self.guards.order_check:
            record.order_hash, record.order_report = self._order_report()
        if self.guards.trial_reversibility:
            record.reversibility_report = self._trial_reversibility(
                record, pre_sol, pre_out, post_out
            )
        self.check_purity()
        if not record.locality_report["pass"]:
            self._violate("locality", record.locality_report, record, rollback=True)
        if record.reversibility_report and not record.reversibility_report["pass"]:
            self._violate(
                "reversibility", record.reversibility_report, record, rollback=True
            )
        if record.order_report and not record.order_report.get("pass", True):
            self._violate(
                "order_invariance", record.order_report, record, rollback=True
            )
        self._remember_state(*post_out)
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
        scores, labels = self._canary_outputs()
        rep = {
            "edit": edit_id,
            "kind": record.kind,
            "downdate": self._recompute_report(),
            "state_seen_before": st.digest in self._seen,
        }
        rep["pass"] = rep["downdate"].get("pass", True)
        if rep["state_seen_before"]:
            cmp = self._compare_outputs(
                self._seen[st.digest], self._fingerprint(scores, labels)
            )
            rep.update(cmp)
            rep["pass"] = (
                rep["pass"]
                and cmp["canary_argmax_identical"]
                and (cmp["canary_max_abs_delta"] <= self.guards.output_tolerance)
            )
        else:
            self._remember_state(scores, labels)
        if not rep["pass"]:
            self._violate("reversibility", rep, None, rollback=False)
        return rep
