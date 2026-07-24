"""Skeleton tests for hybridsim_infer NO_NETWORK path."""

from __future__ import annotations

import unittest

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    build_inference_simulation,
)
from hybridsim_infer.messages import INFER_MESSAGE_TYPES, StepMsg


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
        )
        infra = build_inference_simulation(cfg)
        req = InferenceRequest(
            request_id=42,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=8,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()

        self.assertEqual(infra.cluster.arrived_count, 1)
        self.assertEqual(len(infra.finished_requests), 1)
        done = infra.finished_requests[0]
        self.assertEqual(done.request_id, 42)
        self.assertTrue(done.completed)
        self.assertEqual(done.num_computed_tokens, 16)

    def test_step_msg_registered(self) -> None:
        infra = build_inference_simulation(InferenceConfig(num_replicas=1))
        self.assertIn(StepMsg.__name__, infra.sim.message_types)

    def test_multi_replica_dispatch(self) -> None:
        cfg = InferenceConfig(
            num_replicas=2,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=16,
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


if __name__ == "__main__":
    unittest.main()
