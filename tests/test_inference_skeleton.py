"""Skeleton / KV / schedule tests for hybridsim_infer."""

from __future__ import annotations

import asyncio
import unittest

from hybridsim_infer import (
    BatchFixedConfig,
    BatchLevelConfig,
    ClusterConfig,
    EngineConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    KvConfig,
    KvLookupConfig,
    KvWorkloadConfig,
    ReplicaScheduleConfig,
    ScheduleConfig,
    VllmScheduler,
    build_inference_simulation,
)
from hybridsim_infer.schedulers import SchedulerFactory
from hybridsim_infer.kv_system import VllmKvCacheManager, block_keys_from_tokens
from hybridsim_infer.messages import INFER_MESSAGE_TYPES, StepMsg
from hybridsim_infer.request import RequestStatus


class TestInferenceSkeleton(unittest.TestCase):
    def test_message_registration(self) -> None:
        infra = build_inference_simulation(
            InferenceConfig(cluster=ClusterConfig(num_replicas=1))
        )
        for cls in INFER_MESSAGE_TYPES:
            self.assertIn(cls.__name__, infra.sim.message_types)

    def test_single_request_completes(self) -> None:
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                ),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=4,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)

    def test_multi_replica_round_robinish(self) -> None:
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=2),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        requests = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=4,
                num_decode_tokens=4,
            )
            for i in range(1, 5)
        ]
        infra.schedule_arrivals(requests)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 4)

    def test_engine_busy_does_not_poll_delayed_steps(self) -> None:
        """Worker full → wait BatchEnd; never send delayed StepMsg."""
        dummy_exec_s = 0.1
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(
                    fixed=BatchFixedConfig(dummy_exec_s=dummy_exec_s)
                ),
            ),
        )
        infra = build_inference_simulation(cfg)
        replica = infra.replicas[0]
        orig_send = replica.send
        delayed_steps = []

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            if msg_cls is StepMsg and float(delay) > 0:
                delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        replica.send = counting_send  # type: ignore[method-assign]
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=2,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertEqual(delayed_steps, [])

    def test_chunked_prefill_wakes_on_batch_end(self) -> None:
        """max_inflight=1 chunked prefill still advances after each BatchEnd."""
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=32,
            num_decode_tokens=2,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)

    def test_pipelined_inflight_still_completes(self) -> None:
        """Spare engine slots are filled by same-tick re-arm (max_inflight > 1)."""
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                    max_num_running_reqs=8,
                    max_num_scheduled_tokens=64,
                ),
                engine=EngineConfig(max_inflight_batches=2),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        requests = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=8,
                num_decode_tokens=2,
            )
            for i in range(1, 5)
        ]
        infra.schedule_arrivals(requests)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 4)


class TestSchedulePhases(unittest.TestCase):
    def test_phase1_before_waiting(self) -> None:
        kv = VllmKvCacheManager(num_gpu_blocks=64, block_size=8)
        fw = VllmScheduler(tokens_per_step=4, decode_tokens_per_step=1)
        running = [
            InferenceRequest(
                request_id=1,
                num_prefill_tokens=4,
                num_decode_tokens=4,
                num_computed_tokens=4,
                status=RequestStatus.RUNNING,
            )
        ]
        waiting = [
            InferenceRequest(
                request_id=2,
                num_prefill_tokens=8,
                num_decode_tokens=0,
                status=RequestStatus.WAITING,
            )
        ]
        result = asyncio.run(
            fw.schedule_step(
                waiting,
                running,
                kv_cache_manager=kv,
                batch_id=1,
                token_budget=1,
                max_num_running_reqs=8,
            )
        )
        self.assertIsNotNone(result.batch)
        # Budget exhausted by running decode (1 token) → waiting not admitted.
        self.assertEqual(result.batch.tokens_per_request.get(1), 1)
        self.assertNotIn(2, result.batch.tokens_per_request)
        self.assertEqual(len(result.waiting), 1)

    def test_preemption_on_oom(self) -> None:
        # 2 physical blocks → 1 null reserved → 1 allocatable.
        kv = VllmKvCacheManager(num_gpu_blocks=2, block_size=16)
        fw = VllmScheduler(
            tokens_per_step=16,
            decode_tokens_per_step=16,
            long_prefill_token_threshold=0,
        )
        r1 = InferenceRequest(
            request_id=1,
            num_prefill_tokens=0,
            num_decode_tokens=16,
            num_computed_tokens=0,
            status=RequestStatus.RUNNING,
        )
        r2 = InferenceRequest(
            request_id=2,
            num_prefill_tokens=0,
            num_decode_tokens=16,
            num_computed_tokens=0,
            status=RequestStatus.RUNNING,
        )
        # Allocate r1 first so free_blocks=0.
        self.assertIsNotNone(kv.allocate(r1, 16))
        running = [r1, r2]
        (
            _running_out,
            _scheduled,
            _tokens,
            _blocks,
            preempted,
            _budget,
        ) = fw.process_running_queue(
            running,
            kv_cache_manager=kv,
            token_budget=32,
        )
        self.assertTrue(preempted)
        self.assertEqual(preempted[0].request_id, 2)

    def test_scheduler_factory_vllm(self) -> None:
        fw = SchedulerFactory.create("vllm", tokens_per_step=8)
        self.assertIsInstance(fw, VllmScheduler)
        self.assertEqual(fw.name, "vllm")
        self.assertIn("vllm", SchedulerFactory.registered())

    def test_store_hits_continue_not_break(self) -> None:
        """One schedule() may queue multiple async remote pulls (vLLM continue)."""
        kv = VllmKvCacheManager(num_gpu_blocks=64, block_size=8)
        fw = VllmScheduler(tokens_per_step=8)
        prompt = list(range(8))
        waiting = [
            InferenceRequest(
                request_id=i,
                num_prefill_tokens=8,
                num_decode_tokens=0,
                prompt_token_ids=list(prompt),
                status=RequestStatus.WAITING,
            )
            for i in range(1, 4)
        ]

        def lookup(_req: InferenceRequest) -> dict:
            return {"hit": True, "num_tokens": 8}

        result = asyncio.run(
            fw.schedule_step(
                waiting,
                [],
                kv_cache_manager=kv,
                batch_id=1,
                token_budget=64,
                max_num_running_reqs=32,
                remote_lookup=lookup,
            )
        )
        self.assertEqual(len(result.remote_pulls), 3)
        self.assertIsNone(result.batch)
        self.assertEqual(
            [r.status for r in result.waiting],
            [RequestStatus.WAIT_FOR_REMOTE_KVS] * 3,
        )

    def test_apc_attach_failure_dumps_and_raises(self) -> None:
        prompt = list(range(32))
        kv = VllmKvCacheManager(
            num_gpu_blocks=64, block_size=16, enable_prefix_caching=True
        )
        kv.cache_prefix(prompt)
        fw = VllmScheduler(tokens_per_step=16, enable_prefix_caching=True)
        req = InferenceRequest(
            request_id=1,
            num_prefill_tokens=32,
            num_decode_tokens=0,
            prompt_token_ids=list(prompt),
            status=RequestStatus.WAITING,
        )
        kv.attach_cached_prefix = lambda *_a, **_k: None  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(
                fw.schedule_step(
                    [req],
                    [],
                    kv_cache_manager=kv,
                    batch_id=1,
                    token_budget=64,
                    max_num_running_reqs=8,
                )
            )
        msg = str(ctx.exception)
        self.assertIn("attach_cached_prefix", msg)
        self.assertIn("request_id", msg)

    def test_token_proportional_predictor(self) -> None:
        from hybridsim_infer.workload_generators import TokenProportionalPredictor
        from hybridsim_infer.schedule_types import PrefillChunk, ScheduleBatch

        req = InferenceRequest(request_id=1, num_prefill_tokens=10, num_decode_tokens=0)
        batch = ScheduleBatch(
            batch_id=1,
            chunks=[PrefillChunk(request=req, num_tokens=10)],
            requests=[req],
            tokens_per_request={1: 10},
        )
        pred = TokenProportionalPredictor(
            prefill_s_per_token=0.01, decode_s_per_token=0.1, base_s=0.05
        )
        self.assertAlmostEqual(pred.predict(batch), 0.15)


class TestKvClientPath(unittest.TestCase):
    def test_remote_kv_pull_then_finish(self) -> None:
        # One full block (block_size=8) so Mooncake-style hit is non-zero.
        prompt = list(range(10, 18))
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(enable_store=True, block_size=8),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)

        req = InferenceRequest(
            request_id=7,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=4,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()

        self.assertEqual(len(infra.finished_requests), 1)
        done = infra.finished_requests[0]
        self.assertTrue(done.completed)
        self.assertEqual(done.num_computed_tokens, len(prompt) + 4)

    def test_save_then_second_request_hits(self) -> None:
        prompt = list(range(20, 28))
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(enable_store=True, block_size=8),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        r1 = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=1,
            prompt_token_ids=list(prompt),
        )
        r2 = InferenceRequest(
            request_id=2,
            arrived_at=0.5,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=1,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([r1, r2])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 2)
        # Store should retain at least one block from r1's save.
        assert infra.kv_store is not None
        lookup = infra.kv_store._lookup_keys(block_keys_from_tokens(prompt, 8))
        self.assertTrue(lookup["hit"])
        self.assertEqual(lookup["num_tokens"], 8)

    def test_async_lookup_then_pull(self) -> None:
        prompt = list(range(30, 38))
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                lookup=KvLookupConfig(async_=True, rtt_s=0.005),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)
        req = InferenceRequest(
            request_id=9,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=2,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)
        self.assertEqual(
            infra.finished_requests[0].num_computed_tokens, len(prompt) + 2
        )

    def test_same_tick_drain_schedules_miss_during_pull(self) -> None:
        """Store-hit A pulling must not block miss B; no delayed StepMsg."""
        hit_prompt = list(range(70, 78))
        miss_prompt = list(range(80, 88))
        pull_s = 0.1
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            kv=KvConfig(enable_store=True, block_size=8),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(hit_prompt)
        replica = infra.replicas[0]
        orig_send = replica.send
        delayed_steps: list[float] = []

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            if msg_cls is StepMsg and float(delay) > 0:
                delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        replica.send = counting_send  # type: ignore[method-assign]
        req_a = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(hit_prompt),
            num_decode_tokens=0,
            prompt_token_ids=list(hit_prompt),
        )
        req_b = InferenceRequest(
            request_id=2,
            arrived_at=0.0,
            num_prefill_tokens=len(miss_prompt),
            num_decode_tokens=0,
            prompt_token_ids=list(miss_prompt),
        )
        infra.schedule_arrivals([req_a, req_b])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 2)
        self.assertEqual(delayed_steps, [])
        by_id = {int(r.request_id): r for r in infra.finished_requests}
        self.assertLess(float(by_id[2].finished_at), pull_s)
        self.assertGreater(float(by_id[1].finished_at), float(by_id[2].finished_at))

    def test_wait_for_remote_noop_does_not_rearm(self) -> None:
        """GPU full of WAIT_FOR_REMOTE_KVS: empty steps stay finite; TransferEnd wakes."""
        prompt = list(range(90, 98))
        pull_s = 0.05
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                num_gpu_blocks=2,
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    reserve_full_isl=True,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)
        replica = infra.replicas[0]
        orig_send = replica.send
        delayed_steps: list[float] = []
        step_sends = 0

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            nonlocal step_sends
            if msg_cls is StepMsg:
                step_sends += 1
                if float(delay) > 0:
                    delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        replica.send = counting_send  # type: ignore[method-assign]
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=len(prompt),
                num_decode_tokens=0,
                prompt_token_ids=list(prompt),
            )
            for i in (1, 2)
        ]
        infra.schedule_arrivals(reqs)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 2)
        self.assertEqual(delayed_steps, [])
        # Polling at 1e-4 over two 0.05s pulls would be hundreds of Steps.
        self.assertLess(step_sends, 40)

    def test_kv_contention_deadlock_raises(self) -> None:
        """Partial Store hits fill HBM; after pulls, WAITING hold pages and can_fit fails."""
        # Store seeds one full block (8 tokens). Each request needs 16 tokens total,
        # so after pull they still need one more block — but both already hold one
        # and free_blocks==0 → FCFS can_fit fails with no inflight I/O.
        hit_prompt = list(range(300, 308))
        full_prompt = hit_prompt + list(range(400, 408))
        pull_s = 0.02
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    reserve_full_isl=True,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                num_gpu_blocks=3,
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(hit_prompt)
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=len(full_prompt),
                num_decode_tokens=0,
                prompt_token_ids=list(full_prompt),
            )
            for i in (1, 2)
        ]
        infra.schedule_arrivals(reqs)
        infra.run()
        with self.assertRaises(RuntimeError) as ctx:
            infra.check_errors()
        msg = str(ctx.exception)
        self.assertIn("KV cache contention deadlock", msg)
        self.assertIn("Waiting", msg)
        self.assertLess(len(infra.finished_requests), 2)

    def test_store_hit_interrupt_drains_all_waiting(self) -> None:
        """n same-tick Store hits: continue queues pulls; waiting must empty."""
        prompt = list(range(200, 208))
        n = 8
        pull_s = 0.05
        last_arrival = 0.0
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                num_gpu_blocks=2,
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    reserve_full_isl=True,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)
        replica = infra.replicas[0]
        orig_send = replica.send
        delayed_steps: list[float] = []
        orig_submit = replica._kv.submit_remote_pulls
        n_pulls = 0

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            if msg_cls is StepMsg and float(delay) > 0:
                delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        def counting_pulls(pulls):
            nonlocal n_pulls
            n_pulls += len(pulls)
            return orig_submit(pulls)

        replica.send = counting_send  # type: ignore[method-assign]
        replica._kv.submit_remote_pulls = counting_pulls  # type: ignore[method-assign]
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=last_arrival,
                num_prefill_tokens=len(prompt),
                num_decode_tokens=0,
                prompt_token_ids=list(prompt),
            )
            for i in range(1, n + 1)
        ]
        infra.schedule_arrivals(reqs)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), n)
        self.assertEqual(n_pulls, n)
        self.assertEqual(replica.waiting, [])
        self.assertFalse(replica._step_armed)
        self.assertEqual(delayed_steps, [])
        self.assertGreater(float(infra.now), last_arrival)
        for done in infra.finished_requests:
            self.assertTrue(done.completed)

    def test_many_store_hits_do_not_spin_steps(self) -> None:
        """Tiny HBM + many same-tick Store hits must not dump delay=0 StepMsgs."""
        prompt = list(range(210, 218))
        n = 16
        pull_s = 0.01
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    reserve_full_isl=True,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                num_gpu_blocks=2,
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.005)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)
        replica = infra.replicas[0]
        orig_send = replica.send
        delayed_steps: list[float] = []
        step_sends = 0

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            nonlocal step_sends
            if msg_cls is StepMsg:
                step_sends += 1
                if step_sends > 200:
                    raise AssertionError(
                        f"idle-spin: {step_sends} StepMsg (n={n})"
                    )
                if float(delay) > 0:
                    delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        replica.send = counting_send  # type: ignore[method-assign]
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=len(prompt),
                num_decode_tokens=0,
                prompt_token_ids=list(prompt),
            )
            for i in range(1, n + 1)
        ]
        infra.schedule_arrivals(reqs)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), n)
        self.assertEqual(delayed_steps, [])
        # Serialized pull+compute is O(n) steps, not O(n^2) credit dumps.
        self.assertLess(step_sends, 8 * n)

    def test_wide_hbm_store_hits_pull_in_one_schedule(self) -> None:
        """Enough GPU pages: first step queues a pull per waiting Store hit."""
        prompt = list(range(220, 228))
        n = 6
        pull_s = 0.02
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    reserve_full_isl=True,
                ),
                engine=EngineConfig(max_inflight_batches=1),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                num_gpu_blocks=64,
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.005)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=pull_s,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        infra.kv_store.seed(prompt)
        replica = infra.replicas[0]
        orig_submit = replica._kv.submit_remote_pulls
        pull_batch_sizes: list[int] = []
        orig_send = replica.send
        step_sends = 0
        delayed_steps: list[float] = []
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=0.0,
                num_prefill_tokens=len(prompt),
                num_decode_tokens=0,
                prompt_token_ids=list(prompt),
            )
            for i in range(1, n + 1)
        ]

        def counting_pulls(pulls):
            pull_batch_sizes.append(len(pulls))
            return orig_submit(pulls)

        def counting_send(msg_cls, *, delay: float = 0.0, priority: int = 3, **kwargs):
            nonlocal step_sends
            if msg_cls is StepMsg:
                step_sends += 1
                if step_sends > 200:
                    raise AssertionError(f"idle-spin: {step_sends} StepMsg")
                if float(delay) > 0:
                    delayed_steps.append(float(delay))
            return orig_send(msg_cls, delay=delay, priority=priority, **kwargs)

        def before_run() -> None:
            for req in reqs:
                req.status = RequestStatus.WAITING
                replica.waiting.append(req)
            replica._arm_step()

        replica._kv.submit_remote_pulls = counting_pulls  # type: ignore[method-assign]
        replica.send = counting_send  # type: ignore[method-assign]
        infra.sim.before_run = before_run
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), n)
        self.assertEqual(delayed_steps, [])
        self.assertEqual(sum(pull_batch_sizes), n)
        self.assertGreaterEqual(max(pull_batch_sizes or [0]), n)
        self.assertLess(step_sends, 8 * n)

    def test_pd_handoff_decode_rdma(self) -> None:
        prompt = list(range(40, 48))
        cfg = InferenceConfig(
            cluster=ClusterConfig(
                type="pd",
                num_prefill_replicas=1,
                num_decode_replicas=1,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                lookup=KvLookupConfig(rtt_s=0.002),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        self.assertIsNotNone(infra.kv_store)
        self.assertEqual(len(infra.replicas), 2)
        req = InferenceRequest(
            request_id=11,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=3,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        done = infra.finished_requests[0]
        self.assertTrue(done.completed)
        self.assertEqual(done.num_computed_tokens, len(prompt) + 3)
        params = done.kv_transfer_params or {}
        self.assertTrue(params.get("do_remote_prefill"))
        self.assertTrue(params.get("_handed_off"))
        self.assertEqual(params.get("remote_replica_id"), 0)
        # PD KV pull is not a prefix-cache hit.
        self.assertEqual(done.prefix_hit_tokens, 0)
        self.assertEqual(infra.metrics()["hit_rate"], 0.0)

    def test_pd_prefix_hit_excludes_decode_kv_pull(self) -> None:
        prompt = list(range(16))
        cfg = InferenceConfig(
            cluster=ClusterConfig(
                type="pd",
                num_prefill_replicas=1,
                num_decode_replicas=1,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                    max_num_scheduled_tokens=64,
                ),
            ),
            kv=KvConfig(
                enable_store=True,
                enable_prefix_caching=True,
                block_size=8,
                num_gpu_blocks=256,
                lookup=KvLookupConfig(rtt_s=1e-3),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        infra.schedule_arrivals(
            [
                InferenceRequest(
                    request_id=1,
                    arrived_at=0.0,
                    num_prefill_tokens=len(prompt),
                    num_decode_tokens=2,
                    prompt_token_ids=list(prompt),
                ),
                InferenceRequest(
                    request_id=2,
                    arrived_at=0.2,
                    num_prefill_tokens=len(prompt),
                    num_decode_tokens=2,
                    prompt_token_ids=list(prompt),
                ),
            ]
        )
        infra.run()
        infra.check_errors()
        by_id = {r.request_id: r for r in infra.finished_requests}
        self.assertEqual(len(by_id), 2)
        self.assertEqual(by_id[1].prefix_hit_tokens, 0)
        self.assertGreater(by_id[2].prefix_hit_tokens, 0)
        metrics = infra.metrics()
        self.assertLess(float(metrics["hit_rate"]), 1.0)
        self.assertGreater(float(metrics["hit_rate"]), 0.0)
        self.assertEqual(
            int(metrics["prefix_hit_tokens"]),
            by_id[1].prefix_hit_tokens + by_id[2].prefix_hit_tokens,
        )

    def test_pd_decode_skips_store_hash_match(self) -> None:
        """Decode control-plane lookup ignores Store hash hits (still RTT + pull)."""
        prompt = list(range(50, 58))
        cfg = InferenceConfig(
            cluster=ClusterConfig(
                type="pd",
                num_prefill_replicas=1,
                num_decode_replicas=1,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(
                enable_store=True,
                block_size=8,
                lookup=KvLookupConfig(rtt_s=0.003),
            ),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
            kv_workload=KvWorkloadConfig(
                bandwidth_gbps=100.0,
                transfer_s_floor=1e-4,
            ),
        )
        infra = build_inference_simulation(cfg)
        assert infra.kv_store is not None
        # Seed Store with the same prompt so a hash-match path would also hit.
        infra.kv_store.seed(prompt)
        req = InferenceRequest(
            request_id=12,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=2,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([req])
        t0 = infra.now
        infra.run()
        infra.check_errors()
        done = infra.finished_requests[0]
        self.assertTrue(done.completed)
        # Control-plane RTT must advance sim time beyond a zero-delay hash hit.
        self.assertGreaterEqual(infra.now - t0, cfg.kv.lookup.rtt_s)
        params = done.kv_transfer_params or {}
        self.assertEqual(params.get("remote_replica_id"), 0)

    def test_pd_2p2d_least_load(self) -> None:
        from hybridsim_infer.cluster import PdClusterManager

        mgr = PdClusterManager(
            prefill_replica_ids=[0, 1],
            decode_replica_ids=[2, 3],
        )
        mgr.bind_replicas([object() for _ in range(4)])
        arrive_ids = []
        for i in range(4):
            req = InferenceRequest(
                request_id=i,
                num_prefill_tokens=8,
                num_decode_tokens=1,
            )
            arrive_ids.append(mgr.on_arrive(req))
        self.assertEqual(arrive_ids, [0, 1, 0, 1])

        decode_ids = []
        for i, from_p in enumerate(arrive_ids):
            req = InferenceRequest(
                request_id=100 + i,
                num_prefill_tokens=8,
                num_decode_tokens=1,
            )
            decode_ids.append(
                mgr.on_handoff(req, from_replica_id=from_p, transfer_id=f"t{i}")
            )
            self.assertTrue(req.kv_transfer_params.get("do_remote_prefill"))
            self.assertEqual(req.kv_transfer_params.get("remote_replica_id"), from_p)
        self.assertEqual(decode_ids, [2, 3, 2, 3])

        # Full DES: 2P+2D cluster completes multiple requests.
        cfg = InferenceConfig(
            cluster=ClusterConfig(
                type="pd",
                num_prefill_replicas=2,
                num_decode_replicas=2,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(enable_store=True, block_size=8),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        self.assertEqual(len(infra.replicas), 4)
        prompt = list(range(60, 68))
        reqs = [
            InferenceRequest(
                request_id=i,
                arrived_at=float(i) * 0.01,
                num_prefill_tokens=len(prompt),
                num_decode_tokens=1,
                prompt_token_ids=list(prompt),
            )
            for i in range(4)
        ]
        infra.schedule_arrivals(reqs)
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 4)
        for done in infra.finished_requests:
            self.assertTrue(done.completed)

    def test_local_prefix_reuse(self) -> None:
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=16),
            ),
            kv=KvConfig(enable_prefix_caching=True),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        r1 = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=2,
            prompt_token_ids=list(prompt),
        )
        r2 = InferenceRequest(
            request_id=2,
            arrived_at=0.1,
            num_prefill_tokens=8,
            num_decode_tokens=2,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([r1, r2])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 2)


if __name__ == "__main__":
    unittest.main()
