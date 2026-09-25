"""Evaluation: metrics, the S0 measurement contract (`schema`), paired statistics.

`pal_moe.evaluation` moved here in the v3 restructure (the old name is an alias shim
to the same module objects); `stats` holds the small-sample paired statistics moved
from `experiments/s11_confirmatory.py`.
"""

from .diagnostics import print_router_diagnostics, router_diagnostics
from .geometry import geometry_report, nearest_other_margin, silhouette_score
from .metrics import BenchmarkResult, ContinualEvaluator
from .stats import holm, paired_stats, signed_rank_statistic, tost, westfall_young

__all__ = [
    "holm",
    "paired_stats",
    "signed_rank_statistic",
    "tost",
    "westfall_young",
    "BenchmarkResult",
    "ContinualEvaluator",
    "geometry_report",
    "nearest_other_margin",
    "silhouette_score",
    "router_diagnostics",
    "print_router_diagnostics",
]
