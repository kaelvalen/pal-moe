"""Import shims for modules moved into `pal_moe.legacy`.

A shim does not copy or re-export: it makes the old dotted name *be* the new module
object (`sys.modules[old] is new`), for the package and for every submodule. So
`pal_moe.models.expert.MLPExpert is pal_moe.legacy.models.expert.MLPExpert`, a
monkeypatch through either name is seen through both, and no module is ever loaded
twice under two names (which would silently duplicate classes and registries).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def alias_module(old: str, new: str):
    module = importlib.import_module(new)
    sys.modules[old] = module
    return module


def alias_package(old: str, new: str):
    package = importlib.import_module(new)
    for info in pkgutil.iter_modules(package.__path__):
        sys.modules[f"{old}.{info.name}"] = importlib.import_module(f"{new}.{info.name}")
    sys.modules[old] = package
    return package
