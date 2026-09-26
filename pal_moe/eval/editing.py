"""Knowledge-editing evaluation harness for the v3 LM path (phase 4; no claims).

Loaders read the standard public JSON releases from a local path (nothing is
downloaded here) into one record type, and the metrics are the standard ones:

    CounterFact (ROME, Meng et al. 2022)   efficacy, paraphrase, specificity (neighbourhood)
    zsRE (MEND split, Mitchell et al.)     efficacy, paraphrase, specificity (token accuracy)
    MQuAKE (Zhong et al. 2023)             multi-hop accuracy after the case's edits
    canary set                             locality of the full predict path

Every metric takes a *scorer* - `logprob(prompts, targets) -> tensor` and optionally
`generate(prompts) -> list[str]` - so the same code evaluates the base model, the fast
path (retrieval in context) and the medium path (closed-form edit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from pal_moe.core.hashing import digest, file_digest


@dataclass
class EditCase:
    case_id: str
    prompt: str  # the rewrite prompt, subject filled in
    target_new: str
    target_true: str | None = None
    paraphrases: list[str] = field(default_factory=list)
    neighborhood: list[str] = field(
        default_factory=list
    )  # prompts whose answer must NOT move
    neighborhood_answers: list[str] = field(default_factory=list)  # zsRE loc answers
    source: str = ""


@dataclass
class MultiHopCase:
    case_id: str
    edits: list[EditCase]
    questions: list[str]
    new_answer: str
    new_aliases: list[str] = field(default_factory=list)
    old_answer: str | None = None


def _read(path: str | Path):
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _space(s: str) -> str:
    return s if s.startswith(" ") else " " + s


def load_counterfact(path: str | Path, limit: int | None = None) -> list[EditCase]:
    out = []
    for row in _read(path)[:limit]:
        rw = row["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        out.append(
            EditCase(
                case_id=str(row.get("case_id", len(out))),
                prompt=prompt,
                target_new=_space(rw["target_new"]["str"]),
                target_true=_space(rw["target_true"]["str"]),
                paraphrases=list(row.get("paraphrase_prompts", [])),
                neighborhood=list(row.get("neighborhood_prompts", [])),
                source="counterfact",
            )
        )
    return out


def load_zsre(path: str | Path, limit: int | None = None) -> list[EditCase]:
    out = []
    for i, row in enumerate(_read(path)[:limit]):
        loc_ans = row.get("loc_ans")
        out.append(
            EditCase(
                case_id=str(row.get("case_id", i)),
                prompt=row["src"],
                target_new=_space(row["alt"]),
                target_true=_space(row["answers"][0]) if row.get("answers") else None,
                paraphrases=[row["rephrase"]] if row.get("rephrase") else [],
                neighborhood=[row["loc"]] if row.get("loc") else [],
                neighborhood_answers=[_space(loc_ans)] if loc_ans else [],
                source="zsre",
            )
        )
    return out


def load_mquake(path: str | Path, limit: int | None = None) -> list[MultiHopCase]:
    out = []
    for i, row in enumerate(_read(path)[:limit]):
        edits = [
            EditCase(
                case_id=f"{row.get('case_id', i)}.{j}",
                prompt=rw["prompt"].format(rw["subject"]),
                target_new=_space(rw["target_new"]["str"]),
                target_true=_space(rw["target_true"]["str"]),
                source="mquake",
            )
            for j, rw in enumerate(row["requested_rewrite"])
        ]
        out.append(
            MultiHopCase(
                case_id=str(row.get("case_id", i)),
                edits=edits,
                questions=list(row["questions"]),
                new_answer=row["new_answer"],
                new_aliases=list(row.get("new_answer_alias", [])),
                old_answer=row.get("answer"),
            )
        )
    return out


def load_canary(
    path: str | Path | None = None, prompts: list[str] | None = None
) -> tuple[list[str], str]:
    """A fixed canary prompt set and its content hash (recorded in every run record)."""
    if prompts is None:
        rows = (
            _read(path)
            if str(path).endswith((".json", ".jsonl"))
            else Path(path).read_text().splitlines()
        )
        prompts = [r if isinstance(r, str) else r["prompt"] for r in rows if r]
    return list(prompts), (file_digest(path) if path else digest(prompts))


# -- metrics ---------------------------------------------------------------------------

Scorer = Callable[[list[str], list[str]], torch.Tensor]


def _prefers(logprob: Scorer, prompts: list[str], a: str, b: str) -> list[bool]:
    if not prompts:
        return []
    pa = logprob(prompts, [a] * len(prompts))
    pb = logprob(prompts, [b] * len(prompts))
    return (pa > pb).tolist()


def counterfact_scores(case: EditCase, logprob: Scorer) -> dict:
    """ROME's CounterFact metrics: `P(new) > P(true)` on the rewrite, its paraphrases,
    and (reversed) on the neighbourhood prompts."""
    eff = _prefers(logprob, [case.prompt], case.target_new, case.target_true)
    para = _prefers(logprob, case.paraphrases, case.target_new, case.target_true)
    spec = [
        not v
        for v in _prefers(logprob, case.neighborhood, case.target_new, case.target_true)
    ]
    mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
    return {"efficacy": mean(eff), "paraphrase": mean(para), "specificity": mean(spec)}


def token_accuracy(
    prompts: list[str],
    targets: list[str],
    greedy_next: Callable[[list[str]], list[str]],
) -> float:
    """zsRE-style token accuracy under teacher forcing, via a greedy next-token function."""
    hits = total = 0
    for p, t in zip(prompts, targets):
        prefix = p
        for tok in t:  # character granularity is the caller's tokenizer choice
            hits += int(greedy_next([prefix])[0] == tok)
            total += 1
            prefix += tok
    return hits / max(1, total)


def zsre_scores(
    case: EditCase, logprob: Scorer, base_logprob: Scorer | None = None
) -> dict:
    """Efficacy / paraphrase as mean log-prob gain on the new answer; specificity as the
    neighbourhood answer's log-prob change against the base model (0 = untouched)."""
    out = {"efficacy_logprob": float(logprob([case.prompt], [case.target_new])[0])}
    if case.paraphrases:
        out["paraphrase_logprob"] = float(
            logprob(case.paraphrases, [case.target_new] * len(case.paraphrases)).mean()
        )
    if case.neighborhood and case.neighborhood_answers and base_logprob is not None:
        after = logprob(case.neighborhood, case.neighborhood_answers)
        before = base_logprob(case.neighborhood, case.neighborhood_answers)
        out["specificity_abs_delta"] = float((after - before).abs().mean())
    return out


def multihop_accuracy(
    case: MultiHopCase, generate: Callable[[list[str]], list[str]]
) -> dict:
    """MQuAKE: a case counts if ANY of its questions' generations contains the new answer
    (or an alias), the paper's default aggregation."""
    answers = [case.new_answer, *case.new_aliases]
    gens = generate(case.questions)
    per_q = [any(a.lower() in g.lower() for a in answers) for g in gens]
    return {
        "case_correct": any(per_q),
        "question_accuracy": sum(per_q) / max(1, len(per_q)),
    }


def locality(before: torch.Tensor, after: torch.Tensor) -> dict:
    """Canary locality from next-token logits: argmax flip rate and max |delta logit|."""
    flips = int((before.argmax(-1) != after.argmax(-1)).sum())
    return {
        "flip_rate": flips / max(1, before.size(0)),
        "max_abs_logit_delta": float((after - before).abs().max())
        if before.numel()
        else 0.0,
    }
