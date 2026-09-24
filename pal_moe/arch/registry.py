"""
Name -> implementation registries for the S1 contract.

A registry is the only place that maps a config string ("vit_b_16", "ncm",
"residual_adapter") to a class. Nothing else imports the implementations, so a
new backbone/readout/expert is added by registering it and writing a config,
without touching the training loop (docs/ARCHITECTURE_CONTRACT.md).

Kinds are kept separate on purpose: a readout and an expert have different
signatures, and conflating them is the abstraction error S1 exists to fix.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "Registry",
    "BACKBONES",
    "EXPERTS",
    "CLASSIFICATION_EXPERTS",
    "READOUTS",
    "ROUTERS",
    "register_backbone",
    "register_expert",
    "register_classification_expert",
    "register_readout",
    "register_router",
    "build_backbone",
    "build_expert",
    "build_readout",
    "build_router",
    "describe",
]


class Registry:
    """A minimal, explicit registry: no entry points, no import magic.

    `register` works as a decorator and as a call, and refuses silent
    overwrites so that a name collision is a loud error rather than a
    last-import-wins bug.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        builder: Callable[..., Any] | None = None,
        *,
        replace: bool = False,
    ):
        def _add(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._builders and not replace:
                raise KeyError(
                    f"{self.kind} {name!r} is already registered; pass replace=True "
                    "if the override is intentional"
                )
            self._builders[name] = fn
            return fn

        return _add if builder is None else _add(builder)

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._builders:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._builders[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._builders)

    def __contains__(self, name: object) -> bool:
        return name in self._builders

    def __len__(self) -> int:
        return len(self._builders)


BACKBONES = Registry("backbone")
EXPERTS = Registry("expert")
CLASSIFICATION_EXPERTS = Registry("classification_expert")
READOUTS = Registry("readout")
ROUTERS = Registry("router")

# Thin wrappers so call sites read as `register_backbone("vit_b_16")` rather
# than reaching into the registry object.
register_backbone = BACKBONES.register
register_expert = EXPERTS.register
register_classification_expert = CLASSIFICATION_EXPERTS.register
register_readout = READOUTS.register
register_router = ROUTERS.register


def build_backbone(name: str, **kwargs: Any):
    return BACKBONES.build(name, **kwargs)


def build_expert(name: str, **kwargs: Any):
    return EXPERTS.build(name, **kwargs)


def build_readout(name: str, **kwargs: Any):
    return READOUTS.build(name, **kwargs)


def build_router(name: str, **kwargs: Any):
    return ROUTERS.build(name, **kwargs)


def describe() -> dict[str, list[str]]:
    """The registry contents, for docs and for a result record."""
    return {
        "backbone": BACKBONES.names(),
        "expert": EXPERTS.names(),
        "classification_expert": CLASSIFICATION_EXPERTS.names(),
        "readout": READOUTS.names(),
        "router": ROUTERS.names(),
    }
