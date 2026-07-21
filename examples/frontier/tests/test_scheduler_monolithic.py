"""Integration tests for Frontier bridge on hybridsim actors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

FRONTIER_EXAMPLE = Path(__file__).resolve().parents[1]
if str(FRONTIER_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(FRONTIER_EXAMPLE))

try:
    import frontier  # noqa: F401

    _FRONTIER_AVAILABLE = True
except ImportError:
    _FRONTIER_AVAILABLE = False

if _FRONTIER_AVAILABLE:
    from frontier_bridge import (
        MonolithicConfig,
        ReplicaSchedulerKind,
        build_frontier_simulation,
    )
    from frontier_bridge.simulation_driver import FrontierSimulation


@unittest.skipUnless(_FRONTIER_AVAILABLE, "Frontier not installed (pip install -e $FRONTIER_ROOT)")
class SchedulerMonolithicTests(unittest.TestCase):
    def _run_single_request(self, kind: ReplicaSchedulerKind) -> FrontierSimulation:
        simulation = build_frontier_simulation(
            MonolithicConfig(
                replica_scheduler_kind=kind,
                dummy_execution_time_ms=10.0,
            )
        )
        request = simulation.add_request(
            arrived_at=0.0,
            num_prefill_tokens=4,
            num_decode_tokens=2,
        )
        simulation.inject_requests([request])
        simulation.run()
        simulation.check_errors()
        return simulation

    def test_single_request_vllm_v1_smoke(self) -> None:
        simulation = self._run_single_request(ReplicaSchedulerKind.VLLM_V1)
        self.assertTrue(simulation.all_requests_completed())
        self.assertGreaterEqual(simulation.completed_batches, 1)
        self.assertGreater(simulation.sim.now(), 0.0)

    def test_single_request_sglang_smoke(self) -> None:
        simulation = self._run_single_request(ReplicaSchedulerKind.SGLANG)
        self.assertTrue(simulation.all_requests_completed())
        self.assertGreaterEqual(simulation.completed_batches, 1)

    def test_multi_request_batching(self) -> None:
        simulation = build_frontier_simulation(
            MonolithicConfig(
                replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
                dummy_execution_time_ms=10.0,
            )
        )
        requests = [
            simulation.add_request(
                arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=2
            ),
            simulation.add_request(
                arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=2
            ),
        ]
        simulation.inject_requests(requests)
        simulation.run()
        simulation.check_errors()

        self.assertTrue(all(request.completed for request in requests))
        self.assertGreaterEqual(simulation.completed_batches, 2)

    def test_time_consistency(self) -> None:
        simulation = build_frontier_simulation(
            MonolithicConfig(
                replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
                dummy_execution_time_ms=25.0,
            )
        )
        simulation.inject_requests(
            [
                simulation.add_request(
                    arrived_at=0.0,
                    num_prefill_tokens=4,
                    num_decode_tokens=3,
                )
            ]
        )
        simulation.run()
        simulation.check_errors()

        predicted = simulation.predicted_duration_total
        actual = simulation.sim.now()
        self.assertAlmostEqual(actual, predicted, places=6)

    def test_vllm_v1_vs_sglang_both_complete(self) -> None:
        vllm = self._run_single_request(ReplicaSchedulerKind.VLLM_V1)
        sglang = self._run_single_request(ReplicaSchedulerKind.SGLANG)
        self.assertTrue(vllm.all_requests_completed())
        self.assertTrue(sglang.all_requests_completed())


if __name__ == "__main__":
    unittest.main()
