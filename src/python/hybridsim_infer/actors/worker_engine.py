"""Worker engine: wraps platform EngineActor and reports BatchEndMsg."""

from __future__ import annotations

from typing import Any, Callable, Optional

from hybridsim_infer.schedule_types import ScheduleBatch


class WorkerEngine:
    """Glue between ReplicaScheduler and hybridsim EngineActor.

    Not an ActorBase itself — owns an EngineActor and completion callback.

    In-flight occupancy is owned here: a batch stays counted from ``submit``
    until the replica ``acknowledge``s after processing ``BatchEndMsg``. That
    covers both engine execution and the async completion path, so the replica
    can keep scheduling until ``num_inflight >= max_inflight``.
    """

    def __init__(
        self,
        engine,
        *,
        on_batch_complete: Callable[[int, Optional[ScheduleBatch]], None],
        max_inflight: int = 1,
    ) -> None:
        self._engine = engine
        self._on_batch_complete = on_batch_complete
        self._max_inflight = max(1, int(max_inflight))
        #: workload_id → batch; held until ``acknowledge``.
        self._inflight: dict[int, ScheduleBatch] = {}
        self._engine.set_on_workload_complete(self._handle_complete)

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    @property
    def num_inflight(self) -> int:
        return len(self._inflight)

    @property
    def busy(self) -> bool:
        """True iff at capacity (no more submits allowed)."""
        return self.num_inflight >= self._max_inflight

    def can_submit(self) -> bool:
        return self.num_inflight < self._max_inflight

    def inflight_request_ids(self) -> set[int]:
        ids: set[int] = set()
        for batch in self._inflight.values():
            for req in batch.requests:
                ids.add(int(req.request_id))
        return ids

    def submit(self, workload: dict[str, Any], schedule_batch: ScheduleBatch) -> None:
        if not self.can_submit():
            raise RuntimeError(
                f"WorkerEngine at capacity ({self.num_inflight}/{self._max_inflight})"
            )
        wid = int(workload["workload_id"])
        self._inflight[wid] = schedule_batch
        self._engine.send_workload(workload)

    def acknowledge(self, workload_id: int) -> Optional[ScheduleBatch]:
        """Release occupancy after replica finishes BatchEnd handling."""
        return self._inflight.pop(int(workload_id), None)

    def _handle_complete(self, workload_id: int) -> None:
        wid = int(workload_id)
        batch = self._inflight.get(wid)
        # Keep slot occupied until acknowledge — avoids scheduling before
        # on_batch_complete advances request state.
        self._on_batch_complete(wid, batch)
