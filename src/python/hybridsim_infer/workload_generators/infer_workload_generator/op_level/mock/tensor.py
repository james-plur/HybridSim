"""Activation tensor: shape plus producer op index for dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tensor:
    shape: tuple[int, ...]
    producer: int | None = None
    dtype_bytes: int = 2
