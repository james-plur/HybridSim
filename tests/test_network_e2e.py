"""Small e2e: op_level + FatTree network_sim with TP=2."""

from __future__ import annotations

import unittest

from hybridsim_infer import (
    ClusterConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    NetworkSimConfig,
    ReplicaScheduleConfig,
    ScheduleConfig,
    build_inference_simulation,
)
from hybridsim_infer.workload_generators.configs import ParallelConfig
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_op_level_config,
)


class TestNetworkSimE2E(unittest.TestCase):
    def test_op_level_tp2_completes(self) -> None:
        op_level = resolve_op_level_config(model_preset="llama-3.1-8b")
        op_level.model.num_layers = 1
        op_level.parallel = ParallelConfig(tp_size=2)
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                    max_num_scheduled_tokens=64,
                ),
            ),
            infer_workload=InferWorkloadConfig(mode="op_level", op=op_level),
            network_sim=NetworkSimConfig(
                enabled=True,
                layers=1,
                link_bandwidth_bps=1e12,
                link_delay_s=0.0,
            ),
        )
        infra = build_inference_simulation(cfg)
        self.assertIsNotNone(infra.network)
        self.assertEqual(len(infra.replicas[0]._engines), 2)
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=4,
            num_decode_tokens=1,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)


if __name__ == "__main__":
    unittest.main()
