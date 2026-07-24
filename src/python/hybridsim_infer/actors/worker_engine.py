"""Worker engine: wraps platform EngineActor and reports BatchEndMsg."""

from __future__ import annotations

from typing import Any, Callable, Optional

from hybridsim_infer.messages import BatchEndMsg
from hybridsim_infer.stubs import ScheduleBatch


class WorkerEngine:
    """Glue between ReplicaScheduler and hybridsim EngineActor.

    Not an ActorBase itself — owns an EngineActor and completion callback.
    """

    def __init__(
        self,
        engine,
        *,
        on_batch_complete: Callable[[int, Optional[ScheduleBatch]], None],
    ) -> None:
        self._engine = engine
        self._on_batch_complete = on_batch_complete
        self._inflight: dict[int, ScheduleBatch] = {}
        self._busy = False
        self._engine.set_on_workload_complete(self._handle_complete)

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, workload: dict[str, Any], schedule_batch: ScheduleBatch) -> None:
        wid = int(workload["workload_id"])
        self._inflight[wid] = schedule_batch
        self._busy = True
        self._engine.send_workload(workload)

    def _handle_complete(self, workload_id: int) -> None:
        batch = self._inflight.pop(int(workload_id), None)
        self._busy = bool(self._inflight)
        self._on_batch_complete(int(workload_id), batch)
