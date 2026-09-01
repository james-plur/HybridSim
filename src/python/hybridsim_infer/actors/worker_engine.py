"""Worker engine: wraps platform EngineActor(s) and reports BatchEndMsg."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from hybridsim_infer.config import InferenceConfig
from hybridsim_infer.schedule_types import ScheduleBatch

EngineLike = Any


class WorkerEngine:
    """Glue between ReplicaScheduler and hybridsim EngineActor.

    Not an ActorBase itself — owns one or more EngineActors and a completion
    callback.

    In-flight occupancy is owned here: a batch stays counted from ``submit``
    until the replica ``acknowledge``s after processing ``BatchEndMsg``. That
    covers both engine execution and the async completion path, so the replica
    can keep scheduling until ``num_inflight >= max_inflight``.

    With multiple rank engines, a ``per_rank`` workload is sent to every rank
    and BatchEnd fires only after all ranks report ``WorkloadDoneMsg``.
    """

    def __init__(
        self,
        engine: Union[EngineLike, Sequence[EngineLike]],
        *,
        on_batch_complete: Callable[[int, Optional[ScheduleBatch]], None],
        config: InferenceConfig,
    ) -> None:
        if isinstance(engine, (list, tuple)):
            engines = list(engine)
        else:
            engines = [engine]
        if not engines:
            raise ValueError("WorkerEngine requires at least one EngineActor")
        self._engines = engines
        self._on_batch_complete = on_batch_complete
        self._config = config
        self._max_inflight = max(1, int(config.schedule.engine.max_inflight_batches))
        #: workload_id → batch; held until ``acknowledge``.
        self._inflight: dict[int, ScheduleBatch] = {}
        #: workload_id → ranks still running
        self._pending_ranks: dict[int, set[int]] = {}
        for rank, eng in enumerate(self._engines):
            eng.set_on_workload_complete(
                lambda wid, r=rank: self._handle_complete(r, wid)
            )

    @property
    def engines(self) -> list[EngineLike]:
        return self._engines

    @property
    def engine(self) -> EngineLike:
        return self._engines[0]

    @property
    def num_ranks(self) -> int:
        return len(self._engines)

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

    def start(self) -> None:
        for eng in self._engines:
            eng.start()

    def check_error(self) -> None:
        for eng in self._engines:
            eng.check_error()

    def submit(self, workload: dict[str, Any], schedule_batch: ScheduleBatch) -> None:
        if not self.can_submit():
            raise RuntimeError(
                f"WorkerEngine at capacity ({self.num_inflight}/{self._max_inflight})"
            )
        wid = int(workload["workload_id"])
        self._inflight[wid] = schedule_batch
        per_rank = workload.get("per_rank")
        if per_rank:
            ranks = {int(r) for r in per_rank}
            self._pending_ranks[wid] = set(ranks)
            for rank, wl in per_rank.items():
                r = int(rank)
                kernels = wl["kernels"] if isinstance(wl, dict) else wl
                self._engines[r].send_workload(
                    {"workload_id": wid, "kernels": kernels}
                )
            return
        self._pending_ranks[wid] = {0}
        self._engines[0].send_workload(workload)

    def acknowledge(self, workload_id: int) -> Optional[ScheduleBatch]:
        """Release occupancy after replica finishes BatchEnd handling."""
        self._pending_ranks.pop(int(workload_id), None)
        return self._inflight.pop(int(workload_id), None)

    def _handle_complete(self, rank: int, workload_id: int) -> None:
        wid = int(workload_id)
        pending = self._pending_ranks.get(wid)
        if pending is None:
            return
        pending.discard(int(rank))
        if pending:
            return
        batch = self._inflight.get(wid)
        self._on_batch_complete(wid, batch)
