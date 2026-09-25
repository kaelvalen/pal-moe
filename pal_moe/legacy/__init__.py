"""v1 PAL-MoE modules, behaviour frozen bitwise.

Moved here unchanged in the v3 restructure (phase 1). The old import paths
(`pal_moe.models`, `pal_moe.adaptation`, `pal_moe.baselines`, `pal_moe.builder`,
`pal_moe.trigger`, `pal_moe.memory.prototype_memory`, `pal_moe.memory.generative`,
`pal_moe.factory`, `pal_moe.merge`, `pal_moe.persistence`) are alias shims to the same
module objects (see `_alias.py`), so every published v1 command still runs unchanged.
The guard is the S1 smoke (`run_benchmark.py --dataset mnist --methods naive,palmoe
--epochs 1 --device cpu --seed 42`): identical result JSON before and after the move.
"""
