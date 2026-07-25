"""Compare Mooncake pool profiles (op + hashes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import MooncakePoolEvent, normalize_hash_token


@dataclass
class PoolCompareReport:
    ok: bool
    mismatches: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _event_key(ev: MooncakePoolEvent) -> tuple[Any, ...]:
    hashes = tuple(sorted(normalize_hash_token(h) for h in ev.hashes))
    return (int(ev.step), str(ev.op), hashes, int(ev.num_tokens))


def compare_pool_profiles(
    left: list[MooncakePoolEvent],
    right: list[MooncakePoolEvent],
    *,
    ignore_req_id: bool = True,
) -> PoolCompareReport:
    """Compare profiles by (step, op, sorted hashes, num_tokens)."""
    _ = ignore_req_id
    mismatches: list[str] = []
    if len(left) != len(right):
        mismatches.append(f"length {len(left)} != {len(right)}")
    n = min(len(left), len(right))
    for i in range(n):
        a, b = left[i], right[i]
        ka, kb = _event_key(a), _event_key(b)
        if ka != kb:
            mismatches.append(
                f"event[{i}]: left={ka} right={kb} "
                f"(req_id L={a.req_id!r} R={b.req_id!r})"
            )
    return PoolCompareReport(ok=not mismatches, mismatches=mismatches)
