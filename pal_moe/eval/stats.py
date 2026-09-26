"""Small-sample paired statistics: exact sign / permutation tests, TOST,
Westfall-Young max-T, Holm.

Moved verbatim from `experiments/s11_confirmatory.py` (v3 restructure, phase 1).
"""

import itertools
import math
import statistics


def paired_stats(values: list[float], label: str, sesoi: float | None = None) -> dict:
    """Exact small-sample inference on N paired differences.

    Reports the per-seed distribution first, then mean/SD and a t-based interval
    as a summary, and two *exact* tests: a sign test (only defined away from
    zero) and a paired permutation test over all 2^N sign flips, which is exact
    for any N and handles ties.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"label": label, "n": 0}
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean) if n > 1 else 0.0
    # t-based 95% interval, df = n - 1; a small table rather than a dependency.
    t95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(n - 1, 1.96)
    half = t95 * sd / math.sqrt(n) if n > 1 else 0.0

    positive = sum(1 for v in clean if v > 0)
    negative = sum(1 for v in clean if v < 0)
    nonzero = positive + negative
    sign_p = None
    if nonzero:
        k = min(positive, negative)
        sign_p = min(
            1.0, 2 * sum(math.comb(nonzero, i) for i in range(k + 1)) / 2**nonzero
        )

    # exact paired permutation: every sign flip of the observed differences
    observed = abs(mean)
    extreme = 0
    total = 0
    for flips in itertools.product((1, -1), repeat=n):
        permuted = abs(statistics.mean([f * v for f, v in zip(flips, clean)]))
        extreme += permuted >= observed - 1e-12
        total += 1
    permutation_p = extreme / total if total else None

    out = {
        "label": label,
        "n": n,
        "per_seed": clean,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
        "positive": positive,
        "negative": negative,
        "sign_p": sign_p,
        "permutation_p": permutation_p,
    }
    if sesoi is not None:
        out["sesoi"] = sesoi
        out["within_sesoi"] = bool(abs(mean) + half <= sesoi)
    return out


def signed_rank_statistic(values: list[float]) -> float:
    """Two-sided Wilcoxon statistic: `max(W+, W-)` over the signed ranks.

    Used as the per-test statistic for the max-statistic correction because it is
    scale-free and comparable across tests (unlike a raw mean, which would let
    the largest-scale test dominate) and non-degenerate (unlike a t-statistic,
    which is infinite when the differences are perfectly consistent).

    It must be two-sided: `W+` alone scores a perfectly consistent *negative*
    effect as zero, which made the first run of this stage report the memory
    axis as "not rejected" while every seed agreed on its sign.
    """
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    positive = sum(ranks[i] for i in range(len(values)) if values[i] > 0)
    negative = sum(ranks[i] for i in range(len(values)) if values[i] < 0)
    return max(positive, negative)


def westfall_young(tests: dict[str, list[float]], n_seeds: int) -> dict:
    """Single-step max-statistic (Westfall-Young) correction over a family.

    The tests are paired on the same seeds, so one shared sign-flip permutation
    scheme gives the *joint* null distribution, and the correlation between the
    tests is accounted for instead of paid as a factor of `m`. This matters at
    N = 6: Holm needs p <= alpha/m, and the smallest achievable permutation
    p-value at six seeds is 2/64 = 0.031, so Holm can never reject for any family
    of two or more tests. The max-statistic can.
    """
    names = [
        n for n, v in tests.items() if len(v) == n_seeds and any(v != 0 for v in v)
    ]
    if not names:
        return {}
    observed = {n: signed_rank_statistic(tests[n]) for n in names}
    maxima = []
    for flips in itertools.product((1, -1), repeat=n_seeds):
        flipped = {
            n: signed_rank_statistic([f * v for f, v in zip(flips, tests[n])])
            for n in names
        }
        maxima.append(max(flipped.values()))
    out = {}
    for n in names:
        adjusted = sum(1 for m in maxima if m >= observed[n] - 1e-12) / len(maxima)
        out[n] = {"raw_statistic": observed[n], "adjusted_p": adjusted}
    return out


def tost(values: list[float], sesoi: float) -> dict:
    """Two one-sided tests for equivalence against `+-sesoi`.

    Equivalence is a different question from difference: a directional test can
    only fail to reject, which is not evidence of equivalence. TOST rejects both
    one-sided nulls (`mean <= -sesoi` and `mean >= +sesoi`), which is the same
    as a 90% interval inside the window. It is parametric (t-based) while the
    directional tests above are exact, and the report says so.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n < 2:
        return {"n": n, "note": "TOST needs at least two seeds"}
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean)
    se = sd / math.sqrt(n)
    t_crit = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
    }.get(n - 1, 1.645)

    # one-sided p-values from the t distribution, via the regularized incomplete
    # beta through a small numeric integration (no scipy dependency).
    def t_sf(t, df):
        # P(T > t) for Student-t, computed by numerical integration of the pdf.
        def pdf(x):
            return (
                math.gamma((df + 1) / 2)
                / (math.sqrt(df * math.pi) * math.gamma(df / 2))
                * (1 + x * x / df) ** (-(df + 1) / 2)
            )

        if t <= 0:
            return 1.0
        steps = 20000
        upper = max(t + 50.0, 60.0)
        width = (upper - t) / steps
        total = 0.0
        for i in range(steps):
            x = t + (i + 0.5) * width
            total += pdf(x) * width
        return min(1.0, total)

    t_lower = (mean + sesoi) / se if se > 0 else math.inf
    t_upper = (sesoi - mean) / se if se > 0 else math.inf
    p_lower = t_sf(t_lower, n - 1)
    p_upper = t_sf(t_upper, n - 1)
    return {
        "n": n,
        "mean": mean,
        "sesoi": sesoi,
        "ci90": [mean - t_crit * se, mean + t_crit * se],
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equivalent": bool(p_lower < 0.05 and p_upper < 0.05),
    }


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over the primary family, monotone-adjusted."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for index, (name, p) in enumerate(items):
        value = min(1.0, (m - index) * p)
        running = max(running, value)
        adjusted[name] = {"raw_p": p, "adjusted_p": running, "reject": running < alpha}
    return adjusted
