"""ReplicaSchedulerActor: wait/run queues, Step loop, WorkerEngine glue."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.actors.worker_engine import WorkerEngine
from hybridsim_infer.kv_cache import KvCacheManager
from hybridsim_infer.messages import (
    BatchEndMsg,
    KVTransferEndMsg,
    RequestFinishMsg,
    RequestMsg,
    StepMsg,
)
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.stubs import (
    ScheduleBatch,
    batch,
    inference_workload_generator,
    process_running_queue,
    process_wait_queue,
)


class ReplicaSchedulerActor(ActorBase):
    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        replica_id: int = 0,
        cluster: Any = None,
        engine=None,
        kv_cache_manager: Optional[KvCacheManager] = None,
        step_interval: float = 1e-3,
        dummy_exec_s: float = 0.05,
        tokens_per_step: int = 8,
    ) -> None:
        self.replica_id = replica_id
        self._cluster = cluster
        self._engine = engine
        self._step_interval = float(step_interval)
        self._dummy_exec_s = float(dummy_exec_s)
        self._tokens_per_step = int(tokens_per_step)
        self._kv = kv_cache_manager or KvCacheManager()

        self.waiting: list[InferenceRequest] = []
        self.running: list[InferenceRequest] = []
        self._next_batch_id = 1
        self._next_workload_id = 1
        self._step_armed = False
        self._inflight_batch: Optional[ScheduleBatch] = None

        if engine is None:
            raise ValueError("ReplicaSchedulerActor requires an EngineActor")
        self._worker = WorkerEngine(engine, on_batch_complete=self._on_worker_complete)

        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def start(self) -> None:
        super().start()
        self._engine.start()
        # Kick Step loop once; handler reschedules while there is work.
        self._arm_step()

    def check_error(self) -> None:
        super().check_error()
        self._engine.check_error()

    def _arm_step(self) -> None:
        if self._step_armed:
            return
        self._step_armed = True
        self.send(StepMsg)

    def _has_work(self) -> bool:
        return bool(
            self.waiting or self.running or self._worker.busy or self._inflight_batch
        )

    def _on_worker_complete(
        self, workload_id: int, schedule_batch: Optional[ScheduleBatch]
    ) -> None:
        self.send(BatchEndMsg, workload_id=workload_id, batch=schedule_batch)

    @on(RequestMsg)
    def on_request(self, _actor, msg: RequestMsg) -> None:
        req = msg.request
        req.status = RequestStatus.WAITING
        self.waiting.append(req)
        self._arm_step()

    @on(StepMsg)
    def on_step(self, _actor, msg: StepMsg) -> None:
        self._step_armed = False

        # Do not schedule while a batch is in flight (engine busy or BatchEnd pending).
        if not self._worker.busy and self._inflight_batch is None:
            waiting, admitted, prefill = process_wait_queue(
                self.waiting,
                kv_cache_manager=self._kv,
                tokens_per_step=self._tokens_per_step,
            )
            self.waiting = waiting
            for req in admitted:
                if req not in self.running:
                    self.running.append(req)

            # Newly admitted this step already got a prefill/decode chance in
            # process_wait_queue; continue chunked prefill + decode for the rest.
            admitted_ids = {r.request_id for r in admitted}
            running_for_step = [
                r
                for r in self.running
                if r.status == RequestStatus.RUNNING
                and not r.is_finished()
                and r.request_id not in admitted_ids
            ]
            still_running, more_prefill, decode = process_running_queue(
                running_for_step,
                kv_cache_manager=self._kv,
                tokens_per_step=self._tokens_per_step,
            )
            prefill = list(prefill) + list(more_prefill)
            by_id = {r.request_id: r for r in still_running}
            for r in self.running:
                if r.is_finished():
                    continue
                if r.request_id not in by_id and r.status == RequestStatus.RUNNING:
                    by_id[r.request_id] = r
            self.running = list(by_id.values())

            sched = batch(prefill, decode, batch_id=self._next_batch_id)
            if sched is not None:
                self._next_batch_id += 1
                wid = self._next_workload_id
                self._next_workload_id += 1
                self._inflight_batch = sched
                workload = inference_workload_generator(
                    sched, workload_id=wid, duration_s=self._dummy_exec_s
                )
                self._worker.submit(workload, sched)

        if self._has_work():
            self._step_armed = True
            self.send(StepMsg, delay=self._step_interval)

    @on(BatchEndMsg)
    def on_batch_end(self, _actor, msg: BatchEndMsg) -> None:
        sched: Optional[ScheduleBatch] = msg.batch
        self._inflight_batch = None
        if sched is None:
            self._arm_step()
            return

        finished: list[InferenceRequest] = []
        for req in sched.requests:
            n = int(sched.tokens_per_request.get(req.request_id, 0))
            req.num_computed_tokens += n
            if req.is_finished():
                req.status = RequestStatus.FINISHED
                req.completed = True
                finished.append(req)
                self._kv.free(req)

        if finished:
            finished_ids = {r.request_id for r in finished}
            self.running = [r for r in self.running if r.request_id not in finished_ids]
            if self._cluster is not None:
                for req in finished:
                    self._cluster.send(
                        RequestFinishMsg, request=req, replica_id=self.replica_id
                    )

        self._arm_step()

    @on(KVTransferEndMsg)
    def on_kv_transfer_end(self, _actor, msg: KVTransferEndMsg) -> None:
        # Stub: real path would move WAIT_FOR_REMOTE_KVS → WAITING
        self._arm_step()
