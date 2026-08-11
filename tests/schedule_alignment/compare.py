"""Compare schedule ledgers (step-aligned)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schema import ScheduleStepRecord, filter_nonempty, read_ledger


@dataclass
class StepDiff:
    step: int
    field: str
    left: Any
    right: Any


@dataclass
class CompareReport:
    left_name: str
    right_name: str
    equal: bool
    diffs: list[StepDiff] = field(default_factory=list)
    left_steps: int = 0
    right_steps: int = 0

    def summary(self) -> str:
        if self.equal:
            return (
                f"OK: {self.left_name} == {self.right_name} "
                f"({self.left_steps} nonempty steps)"
            )
        lines = [
            f"MISMATCH: {self.left_name} vs {self.right_name} "
            f"(steps {self.left_steps}/{self.right_steps}, diffs={len(self.diffs)})"
        ]
        for d in self.diffs[:20]:
            lines.append(f"  step={d.step} {d.field}: {d.left!r} != {d.right!r}")
        if len(self.diffs) > 20:
            lines.append(f"  ... and {self.diffs.__len__() - 20} more")
        return "\n".join(lines)


def _norm_int_map(d: dict[str, int] | None) -> dict[str, int]:
    if not d:
        return {}
    return {str(k): int(v) for k, v in d.items()}


def compare_ledgers(
    left: list[ScheduleStepRecord],
    right: list[ScheduleStepRecord],
    *,
    left_name: str = "left",
    right_name: str = "right",
    drop_empty: bool = True,
    compare_queues: bool = False,
    compare_kv: bool = False,
) -> CompareReport:
    a = filter_nonempty(left) if drop_empty else list(left)
    b = filter_nonempty(right) if drop_empty else list(right)
    diffs: list[StepDiff] = []
    n = max(len(a), len(b))
    for i in range(n):
        if i >= len(a):
            diffs.append(StepDiff(i, "missing_left", None, b[i].to_dict()))
            continue
        if i >= len(b):
            diffs.append(StepDiff(i, "missing_right", a[i].to_dict(), None))
            continue
        la, rb = a[i], b[i]
        if la.scheduled_tokens != rb.scheduled_tokens:
            diffs.append(
                StepDiff(i, "scheduled_tokens", la.scheduled_tokens, rb.scheduled_tokens)
            )
        if sorted(la.preempted_ids) != sorted(rb.preempted_ids):
            diffs.append(
                StepDiff(i, "preempted_ids", la.preempted_ids, rb.preempted_ids)
            )
        if sorted(la.finished_ids) != sorted(rb.finished_ids):
            diffs.append(
                StepDiff(i, "finished_ids", la.finished_ids, rb.finished_ids)
            )
        if compare_queues:
            if sorted(la.waiting_ids) != sorted(rb.waiting_ids):
                diffs.append(
                    StepDiff(i, "waiting_ids", la.waiting_ids, rb.waiting_ids)
                )
            if sorted(la.running_ids) != sorted(rb.running_ids):
                diffs.append(
                    StepDiff(i, "running_ids", la.running_ids, rb.running_ids)
                )
        if compare_kv:
            # Missing free_blocks on either side → skip that field (legacy ledgers).
            if la.free_blocks is not None and rb.free_blocks is not None:
                if int(la.free_blocks) != int(rb.free_blocks):
                    diffs.append(
                        StepDiff(i, "free_blocks", la.free_blocks, rb.free_blocks)
                    )
            la_alloc = _norm_int_map(la.allocated_blocks)
            rb_alloc = _norm_int_map(rb.allocated_blocks)
            if la_alloc != rb_alloc:
                diffs.append(
                    StepDiff(i, "allocated_blocks", la_alloc, rb_alloc)
                )
            la_hit = _norm_int_map(la.prefix_hit_tokens)
            rb_hit = _norm_int_map(rb.prefix_hit_tokens)
            if la_hit != rb_hit:
                diffs.append(
                    StepDiff(i, "prefix_hit_tokens", la_hit, rb_hit)
                )
    return CompareReport(
        left_name=left_name,
        right_name=right_name,
        equal=not diffs,
        diffs=diffs,
        left_steps=len(a),
        right_steps=len(b),
    )


def compare_ledger_files(
    left_path: Path,
    right_path: Path,
    **kwargs,
) -> CompareReport:
    return compare_ledgers(
        read_ledger(left_path),
        read_ledger(right_path),
        left_name=str(left_path),
        right_name=str(right_path),
        **kwargs,
    )
