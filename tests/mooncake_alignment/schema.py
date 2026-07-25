"""Mooncake pool profile schema (JSONL) for Store CRUD alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class MooncakePoolEvent:
    """One pool mutation / query event (hybridsim ↔ vLLM Store)."""

    step: int
    op: str  # exist | put | get | delete | evict
    req_id: str = ""
    keys: list[str] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    hit_mask: list[bool] = field(default_factory=list)
    num_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "op": str(self.op),
            "req_id": str(self.req_id),
            "keys": [str(x) for x in self.keys],
            "hashes": [str(x) for x in self.hashes],
            "block_ids": [int(x) for x in self.block_ids],
            "hit_mask": [bool(x) for x in self.hit_mask],
            "num_tokens": int(self.num_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MooncakePoolEvent:
        return cls(
            step=int(data.get("step", 0)),
            op=str(data["op"]),
            req_id=str(data.get("req_id", "")),
            keys=[str(x) for x in (data.get("keys") or [])],
            hashes=[str(x) for x in (data.get("hashes") or [])],
            block_ids=[int(x) for x in (data.get("block_ids") or [])],
            hit_mask=[bool(x) for x in (data.get("hit_mask") or [])],
            num_tokens=int(data.get("num_tokens", 0)),
        )


def write_pool_profile(path: Path, events: Iterable[MooncakePoolEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")


def read_pool_profile(path: Path) -> list[MooncakePoolEvent]:
    path = Path(path)
    rows: list[MooncakePoolEvent] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(MooncakePoolEvent.from_dict(json.loads(line)))
    return rows


def normalize_hash_token(key_or_hash: str) -> str:
    """Prefer bare hash hex (last ``@`` segment of Mooncake PoolKey)."""
    s = str(key_or_hash)
    if "@" in s:
        s = s.rsplit("@", 1)[-1]
    return s.lower()
