"""Moved to `pal_moe.legacy.models` (v3 restructure). Alias shim: same module objects."""

from pal_moe.legacy._alias import alias_package

alias_package(__name__, "pal_moe.legacy.models")
