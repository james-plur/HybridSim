"""Skeleton / KV / schedule tests for hybridsim_infer."""

from __future__ import annotations

import asyncio
import unittest

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    VllmScheduler,
    build_inference_simulation,
)
from hybridsim_infer.schedulers import SchedulerFactory
from hybridsim_infer.kv_system import VllmKvCacheManager, block_keys_from_tokens
from hybridsim_infer.messages import INFER_MESSAGE_TYPES, StepMsg
from hybridsim_infer.request import RequestStatus


class TestInferenceSkeleton(unittest.TestCase):
    def test_message_registration(self) -> None:
        infra = build_inference_simulation(InferenceConfig(num_replicas=1))
        for cls in INFER_MESSAGE_TYPES:
            self.assertIn(cls.__name__, infra.sim.message_types)

    def test_single_request_completes(self) -> None:
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=8,
            decode_tokens_per_step=1,
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
            num_replicas=2,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=8,
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
        """Worker full → wait BatchEnd; do not send StepMsg every step_interval."""
        dummy_exec_s = 0.1
        step_interval = 1e-4
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=step_interval,
            dummy_exec_s=dummy_exec_s,
            tokens_per_step=8,
            decode_tokens_per_step=1,
            max_inflight_batches=1,
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
        # Old loop would enqueue ~dummy_exec_s/step_interval delayed Steps per batch.
        self.assertLess(len(delayed_steps), 20)

    def test_chunked_prefill_wakes_on_batch_end(self) -> None:
        """max_inflight=1 chunked prefill still advances after each BatchEnd."""
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=8,
            decode_tokens_per_step=1,
            max_inflight_batches=1,
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
        """Spare engine slots still use delayed Step to fill max_inflight > 1."""
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=8,
            decode_tokens_per_step=1,
            max_inflight_batches=2,
            max_num_running_reqs=8,
            max_num_scheduled_tokens=64,
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
            num_replicas=1,
            enable_kv_client=True,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            kv_transfer_s=1e-4,
            kv_bandwidth_gbps=100.0,
            block_size=8,
            tokens_per_step=8,
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
            num_replicas=1,
            enable_kv_client=True,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            kv_transfer_s=1e-4,
            kv_bandwidth_gbps=100.0,
            block_size=8,
            tokens_per_step=8,
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
            num_replicas=1,
            enable_kv_client=True,
            kv_lookup_async=True,
            kv_lookup_rtt_s=0.005,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            kv_transfer_s=1e-4,
            kv_bandwidth_gbps=100.0,
            block_size=8,
            tokens_per_step=8,
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

    def test_pd_handoff_decode_rdma(self) -> None:
        prompt = list(range(40, 48))
        cfg = InferenceConfig(
            cluster_type="pd",
            num_prefill_replicas=1,
            num_decode_replicas=1,
            enable_kv_client=True,
            kv_lookup_rtt_s=0.002,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            kv_transfer_s=1e-4,
            kv_bandwidth_gbps=100.0,
            block_size=8,
            tokens_per_step=8,
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

    def test_pd_decode_skips_store_hash_match(self) -> None:
        """Decode control-plane lookup ignores Store hash hits (still RTT + pull)."""
        prompt = list(range(50, 58))
        cfg = InferenceConfig(
            cluster_type="pd",
            num_prefill_replicas=1,
            num_decode_replicas=1,
            enable_kv_client=True,
            kv_lookup_rtt_s=0.003,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            kv_transfer_s=1e-4,
            kv_bandwidth_gbps=100.0,
            block_size=8,
            tokens_per_step=8,
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
        self.assertGreaterEqual(infra.now - t0, cfg.kv_lookup_rtt_s)
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
            cluster_type="pd",
            num_prefill_replicas=2,
            num_decode_replicas=2,
            enable_kv_client=True,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            block_size=8,
            tokens_per_step=8,
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
            num_replicas=1,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=16,
            enable_prefix_caching=True,
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
