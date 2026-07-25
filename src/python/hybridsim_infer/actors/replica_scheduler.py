"""ReplicaSchedulerActor: wait/run queues, Step loop, Worker + optional KV client."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.actors.kv_store import KvClientEngine
from hybridsim_infer.actors.worker_engine import WorkerEngine
from hybridsim_infer.frameworks import FrameworkFactory, InferenceFramework
from hybridsim_infer.kv_cache import KvCacheManager
from hybridsim_infer.messages import (
    BatchEndMsg,
    KVTransferEndMsg,
    KVUpdateMsg,
    RequestFinishMsg,
    RequestMsg,
    StepMsg,
)
from hybridsim_infer.predictors import make_predictor
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.stubs import (
    ScheduleBatch,
    inference_workload_generator,
    kv_transfer_workload,
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
        kv_store: Any = None,
        kv_engine=None,
        framework: Optional[InferenceFramework] = None,
        step_interval: float = 1e-3,
        dummy_exec_s: float = 0.05,
        kv_transfer_s: float = 0.01,
        tokens_per_step: int = 8,
        decode_tokens_per_step: int = 1,
        max_num_scheduled_tokens: int = 64,
        max_num_running_reqs: int = 32,
        long_prefill_token_threshold: int = 0,
        reserve_full_isl: bool = True,
        enable_prefix_caching: bool = False,
        framework_name: str = "vllm",
        duration_mode: str = "fixed",
        prefill_s_per_token: float = 1e-4,
        decode_s_per_token: float = 1e-3,
        duration_base_s: float = 0.0,
        duration_predictor: Any = None,
    ) -> None:
        self.replica_id = replica_id
        self._cluster = cluster
        self._engine = engine
        self._kv_store = kv_store
        self._step_interval = float(step_interval)
        self._dummy_exec_s = float(dummy_exec_s)
        self._kv_transfer_s = float(kv_transfer_s)
        self._max_num_scheduled_tokens = int(max_num_scheduled_tokens)
        self._max_num_running_reqs = int(max_num_running_reqs)
        self._kv = kv_cache_manager or KvCacheManager()
        self._framework = framework or FrameworkFactory.create(
            framework_name,
            tokens_per_step=tokens_per_step,
            decode_tokens_per_step=decode_tokens_per_step,
            long_prefill_token_threshold=long_prefill_token_threshold,
            reserve_full_isl=reserve_full_isl,
            enable_prefix_caching=enable_prefix_caching,
        )
        self._duration_predictor = duration_predictor or make_predictor(
            duration_mode=duration_mode,
            dummy_exec_s=dummy_exec_s,
            prefill_s_per_token=prefill_s_per_token,
            decode_s_per_token=decode_s_per_token,
            base_s=duration_base_s,
        )

        self.waiting: list[InferenceRequest] = []
        self.running: list[InferenceRequest] = []
        self._next_batch_id = 1
        self._next_workload_id = 1
        self._next_kv_workload_id = 1
        self._step_armed = False
        self._inflight_batch: Optional[ScheduleBatch] = None

        if engine is None:
            raise ValueError("ReplicaSchedulerActor requires an EngineActor")
        self._worker = WorkerEngine(engine, on_batch_complete=self._on_worker_complete)

        self._kv_client: Optional[KvClientEngine] = None
        self._pending_kv_pulls: set[int] = set()
        if kv_engine is not None:
            self._kv_client = KvClientEngine(
                kv_engine, on_transfer_complete=self._on_kv_transfer_complete
            )

        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def start(self) -> None:
        super().start()
        self._engine.start()
        if self._kv_client is not None:
            self._kv_client.start()
        self._arm_step()

    def check_error(self) -> None:
        super().check_error()
        self._engine.check_error()
        if self._kv_client is not None:
            self._kv_client.check_error()

    def _arm_step(self) -> None:
        if self._step_armed:
            return
        self._step_armed = True
        self.send(StepMsg)

    def _has_work(self) -> bool:
        kv_busy = self._kv_client.busy if self._kv_client is not None else False
        return bool(
            self.waiting
            or self.running
            or self._worker.busy
            or self._inflight_batch
            or kv_busy
        )

    def _on_worker_complete(
        self, workload_id: int, schedule_batch: Optional[ScheduleBatch]
    ) -> None:
        self.send(BatchEndMsg, workload_id=workload_id, batch=schedule_batch)

    def _on_kv_transfer_complete(self, _workload_id: int, request_id: int) -> None:
        self.send(KVTransferEndMsg, request_id=request_id)

    def _remote_lookup(self, request: InferenceRequest) -> dict[str, Any]:
        assert self._kv_store is not None
        return self._kv_store.lookup(list(request.prompt_token_ids))

    def _submit_remote_pulls(self, pulls) -> None:
        if self._kv_client is None:
            return
        for pull in pulls:
            self._pending_kv_pulls.add(pull.request.request_id)
            wid = self._next_kv_workload_id
            self._next_kv_workload_id += 1
            workload = kv_transfer_workload(
                workload_id=wid,
                request_id=pull.request.request_id,
                duration_s=self._kv_transfer_s,
            )
            self._kv_client.submit(workload, pull.request.request_id)

    @on(RequestMsg)
    def on_request(self, _actor, msg: RequestMsg) -> None:
        req = msg.request
        req.status = RequestStatus.WAITING
        self.waiting.append(req)
        self._arm_step()

    @on(StepMsg)
    def on_step(self, _actor, msg: StepMsg) -> None:
        self._step_armed = False

        if not self._worker.busy and self._inflight_batch is None:
            remote_lookup = (
                self._remote_lookup if self._kv_store is not None else None
            )
            result = self._framework.schedule_step(
                self.waiting,
                self.running,
                kv_cache_manager=self._kv,
                batch_id=self._next_batch_id,
                token_budget=self._max_num_scheduled_tokens,
                max_num_running_reqs=self._max_num_running_reqs,
                remote_lookup=remote_lookup,
            )
            self.waiting = result.waiting
            self.running = result.running

            if result.finished_cached and self._cluster is not None:
                for req in result.finished_cached:
                    self._cluster.send(
                        RequestFinishMsg, request=req, replica_id=self.replica_id
                    )

            if result.remote_pulls:
                self._submit_remote_pulls(result.remote_pulls)

            if result.batch is not None:
                self._next_batch_id += 1
                wid = self._next_workload_id
                self._next_workload_id += 1
                self._inflight_batch = result.batch
                duration_s = float(self._duration_predictor.predict(result.batch))
                workload = inference_workload_generator(
                    result.batch,
                    workload_id=wid,
                    duration_s=duration_s,
                )
                self._worker.submit(workload, result.batch)

        if self._has_work():
            self._step_armed = True
            self.send(StepMsg, delay=self._step_interval)

    @on(BatchEndMsg)
    async def on_batch_end(self, _actor, msg: BatchEndMsg) -> None:
        sched: Optional[ScheduleBatch] = msg.batch
        self._inflight_batch = None
        if sched is None:
            self._arm_step()
            return

        finished = self._framework.on_batch_complete(
            list(sched.requests),
            sched.tokens_per_request,
            self._kv,
        )

        # Optional: push computed prefixes to remote KV store.
        if self._kv_store is not None:
            for req in sched.requests:
                prefix = list(req.prompt_token_ids[: req.num_computed_tokens])
                if not prefix:
                    continue
                reply = await self.request(
                    self._kv_store,
                    KVUpdateMsg,
                    token_ids=prefix,
                    request_id=req.request_id,
                )
                if (
                    reply
                    and reply.get("ok")
                    and self._kv_client is not None
                    and not req.completed
                ):
                    wid = self._next_kv_workload_id
                    self._next_kv_workload_id += 1
                    self._kv_client.submit(
                        kv_transfer_workload(
                            workload_id=wid,
                            request_id=req.request_id,
                            duration_s=self._kv_transfer_s,
                        ),
                        req.request_id,
                    )

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
        rid = int(msg.request_id)
        if rid in self._pending_kv_pulls:
            self._pending_kv_pulls.discard(rid)
            for req in self.waiting:
                if req.request_id != rid:
                    continue
                if req.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
                    hit = req.pending_remote_tokens
                    req.num_computed_tokens = max(req.num_computed_tokens, hit)
                    req.pending_remote_tokens = 0
                    req.status = RequestStatus.WAITING
                break
        self._arm_step()
