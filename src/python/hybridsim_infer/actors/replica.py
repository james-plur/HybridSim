"""ReplicaActor: wait/run queues, Step loop, Worker + KV via manager."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.actors.worker_engine import WorkerEngine
from hybridsim_infer.schedulers import SchedulerFactory, InferenceScheduler
from hybridsim_infer.kv_system import KvCacheManager, KvClient, VllmKvCacheManager
from hybridsim_infer.messages import (
    BatchEndMsg,
    KVLookupReplyMsg,
    KVTransferEndMsg,
    RequestFinishMsg,
    RequestHandoffMsg,
    RequestMsg,
    StepMsg,
)
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.workload_generators import (
    WorkloadGenerator,
    make_workload_generator,
)


class ReplicaActor(ActorBase):
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
        scheduler: Optional[InferenceScheduler] = None,
        step_interval: float = 1e-3,
        dummy_exec_s: float = 0.05,
        kv_transfer_s: float = 1e-4,
        kv_bandwidth_gbps: float = 50.0,
        kv_bytes_per_token: float | None = None,
        kv_latency_s: float = 0.0,
        kv_lookup_async: bool = False,
        kv_lookup_rtt_s: float = 1e-3,
        tokens_per_step: int = 8,
        decode_tokens_per_step: int = 1,
        max_num_scheduled_tokens: int = 64,
        max_num_running_reqs: int = 32,
        max_inflight_batches: int = 1,
        long_prefill_token_threshold: int = 0,
        reserve_full_isl: bool = True,
        enable_prefix_caching: bool = False,
        scheduler_name: str = "vllm",
        duration_mode: str = "fixed",
        prefill_s_per_token: float = 1e-4,
        decode_s_per_token: float = 1e-3,
        duration_base_s: float = 0.0,
        duration_predictor: Any = None,
        frontier_predictor: Any = None,
        frontier_cluster_type: Any = None,
        frontier_replica_id: int = 0,
        frontier_is_moe: bool = False,
        analytical_config: Any = None,
        workload_generator: Optional[WorkloadGenerator] = None,
        profile: Any = None,
    ) -> None:
        self.replica_id = replica_id
        self._cluster = cluster
        self._engine = engine
        self._profile = profile
        self._step_interval = float(step_interval)
        self._dummy_exec_s = float(dummy_exec_s)
        self._max_num_scheduled_tokens = int(max_num_scheduled_tokens)
        self._max_num_running_reqs = int(max_num_running_reqs)
        self._kv: KvCacheManager = kv_cache_manager or VllmKvCacheManager()
        self._scheduler = scheduler or SchedulerFactory.create(
            scheduler_name,
            tokens_per_step=tokens_per_step,
            decode_tokens_per_step=decode_tokens_per_step,
            long_prefill_token_threshold=long_prefill_token_threshold,
            reserve_full_isl=reserve_full_isl,
            enable_prefix_caching=enable_prefix_caching,
        )
        self._workload_generator: WorkloadGenerator = (
            workload_generator
            or make_workload_generator(
                duration_mode=duration_mode,
                dummy_exec_s=dummy_exec_s,
                prefill_s_per_token=prefill_s_per_token,
                decode_s_per_token=decode_s_per_token,
                duration_base_s=duration_base_s,
                predictor=duration_predictor,
                frontier_predictor=frontier_predictor,
                frontier_cluster_type=frontier_cluster_type,
                frontier_replica_id=frontier_replica_id,
                frontier_is_moe=frontier_is_moe,
                analytical_config=analytical_config,
            )
        )

        self.waiting: list[InferenceRequest] = []
        self.running: list[InferenceRequest] = []
        self._next_batch_id = 1
        self._next_workload_id = 1
        self._step_armed = False

        if engine is None:
            raise ValueError("ReplicaActor requires an EngineActor")
        self._worker = WorkerEngine(
            engine,
            on_batch_complete=self._on_worker_complete,
            max_inflight=max_inflight_batches,
        )

        # Homogeneous replicas: wire KvClient whenever a transfer engine is provided.
        # Store is optional (monolith/PD prefix pool); PD Decode uses control-plane lookup.
        if kv_engine is not None:
            from hybridsim_infer.workload_generators.analytic_model.configs import (
                NetworkConfig,
            )

            model_cfg = None
            if analytical_config is not None and hasattr(analytical_config, "model"):
                model_cfg = analytical_config.model
            net_cfg = NetworkConfig.from_bandwidth(
                latency_s=float(kv_latency_s),
                bandwidth_gbps=float(kv_bandwidth_gbps),
            )
            client = KvClient(
                self,
                kv_store,
                kv_engine,
                block_size=self._kv.block_size,
                store_block_size=getattr(self._kv, "store_block_size", None),
                bandwidth_gbps=kv_bandwidth_gbps,
                bytes_per_token=(
                    float(kv_bytes_per_token) if kv_bytes_per_token is not None else 16.0
                ),
                transfer_s_floor=kv_transfer_s,
                kv_latency_s=kv_latency_s,
                lookup_rtt_s=kv_lookup_rtt_s,
                on_transfer_complete=self._on_kv_transfer_complete,
                model_config=model_cfg,
                network_config=net_cfg,
                profile=self._profile,
                replica_id=self.replica_id,
            )
            self._kv.enable_prefix_caching = bool(
                self._scheduler.enable_prefix_caching
            )
            self._kv.attach_client(
                client,
                kv_lookup_async=bool(kv_lookup_async),
            )

        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def start(self) -> None:
        super().start()
        self._engine.start()
        self._kv.start_client()
        self._arm_step()

    def check_error(self) -> None:
        super().check_error()
        self._engine.check_error()
        self._kv.check_client_error()

    def _arm_step(self) -> None:
        if self._step_armed:
            return
        self._step_armed = True
        self.send(StepMsg)

    def _has_work(self) -> bool:
        return bool(
            self.waiting
            or self.running
            or self._worker.num_inflight
            or self._kv.client_busy
        )

    def _on_worker_complete(
        self, workload_id: int, schedule_batch: Optional[Any]
    ) -> None:
        self.send(BatchEndMsg, priority=1, workload_id=workload_id, batch=schedule_batch)

    def _on_kv_transfer_complete(
        self, _workload_id: int, request_id: int, direction: str
    ) -> None:
        self.send(KVTransferEndMsg, priority=1, request_id=request_id, direction=direction)

    def _maybe_handoff_prefill(self, req: InferenceRequest) -> bool:
        """If request asks for remote decode and Prefill is done, hand off to Cluster."""
        if self._cluster is None:
            return False
        params = req.kv_transfer_params or {}
        if not params.get("do_remote_decode") or params.get("_handed_off"):
            return False
        if req.num_computed_tokens < req.num_prefill_tokens:
            return False
        params = dict(params)
        params["_handed_off"] = True
        req.kv_transfer_params = params
        req.completed = False
        # Prefill completes before decode tokens: still publish local prefix for reuse.
        if self._scheduler.enable_prefix_caching:
            self._kv.cache_request_prefix(req)
        self._kv.free(req)
        transfer_id = str(params.get("transfer_id", ""))
        self._cluster.send(
            RequestHandoffMsg,
            request=req,
            from_replica_id=self.replica_id,
            transfer_id=transfer_id,
        )
        return True

    @on(RequestMsg)
    def on_request(self, _actor, msg: RequestMsg) -> None:
        req = msg.request
        req.status = RequestStatus.WAITING
        self.waiting.append(req)
        if self._profile is not None:
            self._profile.emit_replica_enqueue(
                time_s=float(self.sim.now()),
                replica_id=self.replica_id,
                request_id=int(req.request_id),
                request=req,
            )
        self._arm_step()

    @on(KVLookupReplyMsg)
    def on_kv_lookup_reply(self, _actor, msg: KVLookupReplyMsg) -> None:
        """Cache async lookup result only; allocate happens on the next schedule step."""
        result = self._kv.on_lookup_reply(msg)
        rid = int(msg.request_id)
        for req in self.waiting:
            if req.request_id != rid:
                continue
            req.lookup_result = result
            req.pending_lookup = False
            break
        self._arm_step()

    @on(StepMsg)
    async def on_step(self, _actor, msg: StepMsg) -> None:
        self._step_armed = False

        # One schedule step → at most one batch; gate on Worker inflight depth.
        if self._worker.can_submit():
            blocked = self._worker.inflight_request_ids()
            waiting = [r for r in self.waiting if r.request_id not in blocked]
            running_held = [r for r in self.running if r.request_id in blocked]
            running_ready = [
                r for r in self.running if r.request_id not in blocked
            ]

            remote_lookup = (
                self._kv.remote_lookup if self._kv.remote_enabled else None
            )
            sched_t0 = float(self.sim.now())
            result = await self._scheduler.schedule_step(
                waiting,
                running_ready,
                kv_cache_manager=self._kv,
                batch_id=self._next_batch_id,
                token_budget=self._max_num_scheduled_tokens,
                max_num_running_reqs=self._max_num_running_reqs,
                remote_lookup=remote_lookup,
            )
            # Preserve requests still executing on the Worker.
            self.waiting = result.waiting
            self.running = running_held + result.running

            batch_req_ids: list[int] = []
            if result.batch is not None:
                batch_req_ids = [int(r.request_id) for r in result.batch.requests]
            if self._profile is not None:
                self._profile.emit_replica_schedule(
                    time_s=sched_t0,
                    replica_id=self.replica_id,
                    batch_id=(
                        int(result.batch.batch_id)
                        if result.batch is not None
                        else None
                    ),
                    request_ids=batch_req_ids or None,
                )

            if result.finished_cached and self._cluster is not None:
                for req in result.finished_cached:
                    if self._maybe_handoff_prefill(req):
                        continue
                    self._cluster.send(
                        RequestFinishMsg, request=req, replica_id=self.replica_id
                    )

            if result.remote_pulls:
                self._kv.submit_remote_pulls(result.remote_pulls)

            if result.batch is not None:
                self._next_batch_id += 1
                wid = self._next_workload_id
                self._next_workload_id += 1
                workload = self._workload_generator(result.batch, workload_id=wid)
                engine_start = float(self.sim.now())
                kernels = workload.get("kernels") or []
                duration_s = float(kernels[0].get("duration", 0.0)) if kernels else 0.0
                if self._profile is not None:
                    for req in result.batch.requests:
                        self._profile.emit_engine_req(
                            start_s=engine_start,
                            duration_s=duration_s,
                            replica_id=self.replica_id,
                            request_id=int(req.request_id),
                            workload_id=wid,
                            batch_id=int(result.batch.batch_id),
                            request=req,
                        )
                self._worker.submit(workload, result.batch)

        if self._has_work():
            self._step_armed = True
            self.send(StepMsg, delay=self._step_interval)

    @on(BatchEndMsg)
    async def on_batch_end(self, _actor, msg: BatchEndMsg) -> None:
        wid = int(msg.workload_id)
        sched = msg.batch
        if sched is None:
            self._worker.acknowledge(wid)
            self._arm_step()
            return

        finished = self._scheduler.on_batch_complete(
            list(sched.requests),
            sched.tokens_per_request,
            self._kv,
        )

        await self._kv.save_computed_prefixes(list(sched.requests))

        handed_off_ids: set[int] = set()
        for req in list(sched.requests):
            if self._maybe_handoff_prefill(req):
                handed_off_ids.add(req.request_id)

        if finished or handed_off_ids:
            drop_ids = {r.request_id for r in finished} | handed_off_ids
            self.running = [r for r in self.running if r.request_id not in drop_ids]
            if self._cluster is not None:
                for req in finished:
                    if req.request_id in handed_off_ids:
                        continue
                    self._cluster.send(
                        RequestFinishMsg, request=req, replica_id=self.replica_id
                    )

        # Release Worker slot only after state is advanced (async occupancy).
        self._worker.acknowledge(wid)
        self._arm_step()

    @on(KVTransferEndMsg)
    def on_kv_transfer_end(self, _actor, msg: KVTransferEndMsg) -> None:
        if str(getattr(msg, "direction", "pull")) != "pull":
            self._arm_step()
            return
        self._kv.apply_pull_complete(int(msg.request_id), self.waiting)
        self._arm_step()
