"""In-process Mooncake pool profile recorder for offline hybridsim drivers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .schema import MooncakePoolEvent, write_pool_profile

_EVENTS: list[MooncakePoolEvent] = []
_STEP: int = 0
_PATH: Optional[Path] = None


def reset(*, path: str | Path | None = None, step: int = 0) -> None:
    global _EVENTS, _STEP, _PATH
    _EVENTS = []
    _STEP = int(step)
    _PATH = Path(path) if path else None


def set_step(step: int) -> None:
    global _STEP
    _STEP = int(step)


def record(
    op: str,
    *,
    hashes: Iterable[str] | None = None,
    keys: Iterable[str] | None = None,
    block_ids: Iterable[int] | None = None,
    hit_mask: Iterable[bool] | None = None,
    num_tokens: int = 0,
    req_id: str = "",
    step: int | None = None,
) -> None:
    _EVENTS.append(
        MooncakePoolEvent(
            step=int(_STEP if step is None else step),
            op=str(op),
            req_id=str(req_id),
            keys=[str(x) for x in (keys or [])],
            hashes=[str(x) for x in (hashes or [])],
            block_ids=[int(x) for x in (block_ids or [])],
            hit_mask=[bool(x) for x in (hit_mask or [])],
            num_tokens=int(num_tokens),
        )
    )


def events() -> list[MooncakePoolEvent]:
    return list(_EVENTS)


def flush(path: str | Path | None = None) -> Path | None:
    out = Path(path) if path else _PATH
    if out is None:
        return None
    write_pool_profile(out, _EVENTS)
    return out
