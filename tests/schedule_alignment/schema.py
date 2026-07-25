"""Schedule ledger schema for hybridsim ↔ vLLM offline alignment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class ScheduleStepRecord:
    """One schedule step (index-aligned across drivers; not wall-clock)."""

    step: int
    scheduled_tokens: dict[str, int] = field(default_factory=dict)
    new_req_ids: list[str] = field(default_factory=list)
    preempted_ids: list[str] = field(default_factory=list)
    finished_ids: list[str] = field(default_factory=list)
    waiting_ids: list[str] = field(default_factory=list)
    running_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Stable key order for scheduled_tokens
        d["scheduled_tokens"] = {
            str(k): int(v) for k, v in sorted(self.scheduled_tokens.items(), key=lambda x: str(x[0]))
        }
        for key in ("new_req_ids", "preempted_ids", "finished_ids", "waiting_ids", "running_ids"):
            d[key] = [str(x) for x in d[key]]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduleStepRecord:
        return cls(
            step=int(data["step"]),
            scheduled_tokens={str(k): int(v) for k, v in (data.get("scheduled_tokens") or {}).items()},
            new_req_ids=[str(x) for x in (data.get("new_req_ids") or [])],
            preempted_ids=[str(x) for x in (data.get("preempted_ids") or [])],
            finished_ids=[str(x) for x in (data.get("finished_ids") or [])],
            waiting_ids=[str(x) for x in (data.get("waiting_ids") or [])],
            running_ids=[str(x) for x in (data.get("running_ids") or [])],
        )


def normalize_req_id(req_id: Any) -> str:
    return str(req_id)


def write_ledger(path: Path, records: Iterable[ScheduleStepRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def read_ledger(path: Path) -> list[ScheduleStepRecord]:
    path = Path(path)
    rows: list[ScheduleStepRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(ScheduleStepRecord.from_dict(json.loads(line)))
    return rows


def filter_nonempty(records: list[ScheduleStepRecord]) -> list[ScheduleStepRecord]:
    """Drop steps with no scheduled tokens and no preempt/finish events."""
    out: list[ScheduleStepRecord] = []
    for r in records:
        if r.scheduled_tokens or r.preempted_ids or r.finished_ids or r.new_req_ids:
            out.append(r)
    # Re-index for comparison convenience
    for i, r in enumerate(out):
        r.step = i
    return out
