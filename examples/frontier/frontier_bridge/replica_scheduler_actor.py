"""Replica-level scheduling actor."""

from __future__ import annotations

from typing import Any, Callable, Optional

from frontier.entities import Batch
from frontier.types import ClusterType

from hybridsim import ActorBase, on
from frontier_bridge.batch_completion import (
    complete_global_batch,
    complete_prefill_batch,
)
from frontier_bridge.batch_executor import BatchWorkload, build_batch_workload
from frontier_bridge.messages import BatchCompleteMsg, ReplicaScheduleMsg
from hybridsim import ScheduleTraceRecorder


class ReplicaSchedulerActor(ActorBase):
    """Runs replica batching logic and drives hybridsim EngineActor execution."""

    def __init__(
        self,
        *,
        sim,
        hs_actor,
        engine,
        replica_scheduler,
        predictor,
        cluster_type: ClusterType,
        cluster_scheduler,
        replica_id: int = 0,
        dp_id: int = 0,
        message_types: dict[str, Any],
        trace: Optional[ScheduleTraceRecorder] = None,
        kv_cache_transfer_predictor: Any = None,
        on_kv_transfer: Optional[Callable[[Any], None]] = None,
        on_schedule_complete: Optional[Callable[[], None]] = None,
        on_reschedule: Optional[Callable[[], None]] = None,
    ) -> None:
        self._engine = engine
        self._replica_scheduler = replica_scheduler
        self._predictor = predictor
        self._cluster_type = cluster_type
        self._cluster_scheduler = cluster_scheduler
        self._replica_id = replica_id
        self._dp_id = dp_id
        self._trace = trace
        self._kv_cache_transfer_predictor = kv_cache_transfer_predictor
        self._on_kv_transfer = on_kv_transfer
        self._on_schedule_complete = on_schedule_complete
        self._on_reschedule = on_reschedule

        self._next_workload_id = 1
        self._inflight: dict[int, BatchWorkload] = {}
        self._kv_inflight: dict[int, Any] = {}
        self._predicted_duration_total = 0.0
        self._completed_batches = 0

        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)
        self._engine.set_on_workload_complete(self._on_engine_workload_complete)

    @property
    def predicted_duration_total(self) -> float:
        return self._predicted_duration_total

    @property
    def completed_batches(self) -> int:
        return self._completed_batches

    @property
    def inflight_count(self) -> int:
        return len(self._inflight) + len(self._kv_inflight)

    def start(self) -> None:
        super().start()
        self._engine.start()

    def check_error(self) -> None:
        super().check_error()
        self._engine.check_error()

    def send_replica_schedule(self) -> None:
        self.send(ReplicaScheduleMsg, cluster_type=self._cluster_type)

    @on(ReplicaScheduleMsg)
    def handle_replica_schedule(self, _actor, _msg) -> None:
        now = self.sim.now()
        if self._trace is not None:
            self._trace.record_instant(
                name="ReplicaSchedule",
                time_s=now,
                cluster_type=self._cluster_type.name,
                replica_id=self._replica_id,
            )
        batches = self._replica_scheduler.on_schedule(now)
        if not batches:
            if self._on_schedule_complete is not None:
                self._on_schedule_complete()
            return

        for batch in batches:
            batch.on_schedule(now, self._cluster_type)
            workload_id = self._next_workload_id
            self._next_workload_id += 1

            batch_workload = build_batch_workload(
                workload_id=workload_id,
                batch=batch,
                predictor=self._predictor,
                cluster_type=self._cluster_type,
            )
            self._inflight[workload_id] = batch_workload
            self._predicted_duration_total += batch_workload.duration

            if self._trace is not None:
                self._trace.record_duration(
                    name=f"batch_{batch.id}",
                    start_s=now,
                    duration_s=batch_workload.duration,
                    cluster_type=self._cluster_type.name,
                    replica_id=self._replica_id,
                    request_ids=[request.id for request in batch.requests],
                    args={
                        "batch_id": batch.id,
                        "num_tokens": list(batch.num_tokens),
                        "phase": "prefill"
                        if batch.num_tokens and max(batch.num_tokens) > 1
                        else "decode",
                    },
                )

            from frontier_bridge.batch_executor import batch_to_workload_dict

            self._engine.send_workload(
                batch_to_workload_dict(
                    workload_id,
                    batch,
                    batch_workload.duration,
                )
            )

    def _on_engine_workload_complete(self, workload_id: int) -> None:
        if workload_id in self._kv_inflight:
            transfer_info = self._kv_inflight.pop(workload_id)
            if self._trace is not None:
                self._trace.record_instant(
                    name="KVCacheTransferEnd",
                    time_s=self.sim.now(),
                    cluster_type="KV_TRANSFER",
                    args={
                        "batch_id": transfer_info.batch.id,
                        "bytes": transfer_info.kv_cache_size_bytes,
                    },
                )
            if self._on_kv_transfer is not None:
                self._on_kv_transfer(transfer_info)
            return

        context = self._inflight.get(workload_id)
        if context is None:
            return

        self.send(
            BatchCompleteMsg,
            workload_id=workload_id,
            cluster_type=self._cluster_type,
            batch_schedule_epoch=context.batch_schedule_epoch,
            request_execution_signatures=context.request_execution_signatures,
            request_mutation_signatures=context.request_mutation_signatures,
            thinking_round_start_times=context.thinking_round_start_times,
        )

    @on(BatchCompleteMsg)
    def handle_batch_complete(self, _actor, msg) -> None:
        context = self._inflight.pop(msg.workload_id, None)
        if context is None:
            return

        batch: Batch = context.batch
        if batch.schedule_epoch != msg.batch_schedule_epoch:
            return

        now = self.sim.now()
        self._completed_batches += 1

        if self._cluster_type == ClusterType.PREFILL:
            plan, self._next_workload_id = complete_prefill_batch(
                time_s=now,
                batch=batch,
                replica_id=self._replica_id,
                dp_id=self._dp_id,
                cluster_scheduler=self._cluster_scheduler,
                kv_cache_transfer_predictor=self._kv_cache_transfer_predictor,
                batch_schedule_epoch=msg.batch_schedule_epoch,
                next_workload_id=self._next_workload_id,
            )
            for transfer in plan.kv_transfers or []:
                self._kv_inflight[transfer.workload_id] = transfer.transfer_info
                if self._trace is not None:
                    self._trace.record_duration(
                        name="KVCacheTransfer",
                        start_s=now,
                        duration_s=transfer.duration_s,
                        cluster_type="KV_TRANSFER",
                        replica_id=self._replica_id,
                        request_ids=[
                            request.id
                            for request in transfer.transfer_info.batch.requests
                        ],
                        args={
                            "bytes": transfer.transfer_info.kv_cache_size_bytes,
                            "target": transfer.transfer_info.target_cluster_type.name,
                        },
                    )
                from frontier_bridge.batch_executor import batch_to_workload_dict

                self._engine.send_workload(
                    batch_to_workload_dict(
                        transfer.workload_id,
                        transfer.transfer_info.batch,
                        transfer.duration_s,
                    )
                )
            if plan.reschedule_replica:
                self.send_replica_schedule()
            return

        complete_global_batch(
            time_s=now,
            batch=batch,
            cluster_type=self._cluster_type,
            replica_id=self._replica_id,
            dp_id=self._dp_id,
            cluster_scheduler=self._cluster_scheduler,
            batch_schedule_epoch=msg.batch_schedule_epoch,
            request_execution_signatures=msg.request_execution_signatures,
            request_mutation_signatures=msg.request_mutation_signatures,
            thinking_round_start_times=msg.thinking_round_start_times,
        )
        self.send_replica_schedule()
