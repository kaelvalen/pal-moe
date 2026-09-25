"""Content hashing for the v3 state: tensors, modules, files, edit logs.

A hash here is a statement about bytes, not about values: two tensors with the
same values but different dtypes hash differently. That is the point - the
reversibility guard is a *bitwise* guard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def _update(h, part) -> None:
    if isinstance(part, torch.Tensor):
        t = part.detach().to("cpu").contiguous()
        h.update(f"tensor:{t.dtype}:{tuple(t.shape)}:".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(part, (bytes, bytearray)):
        h.update(b"bytes:" + bytes(part))
    elif isinstance(part, (list, tuple)):
        h.update(f"seq:{len(part)}:".encode())
        for p in part:
            _update(h, p)
    elif isinstance(part, dict):
        h.update(f"map:{len(part)}:".encode())
        for k in sorted(part, key=str):
            _update(h, str(k))
            _update(h, part[k])
    else:
        h.update(f"{type(part).__name__}:{part!r}".encode())
    h.update(b"|")


def digest(*parts) -> str:
    """sha256 over an ordered sequence of tensors / bytes / str / containers."""
    h = hashlib.sha256()
    for part in parts:
        _update(h, part)
    return h.hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    return digest(t)


def module_digest(module: torch.nn.Module) -> str:
    """Hash of every parameter and buffer, by name."""
    return digest({name: t for name, t in module.state_dict().items()})


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
