"""Integration tests for hybridsim scheduler actors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "build"))

try:
    import frontier  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Frontier is required for scheduler tests. Install with:\n"
        "  pip install -e /path/to/Frontier"
    ) from exc

from hybridsim_scheduler import MonolithicConfig, ReplicaSchedulerKind, Simulation


def _run_single_request(kind: ReplicaSchedulerKind) -> Simulation:
    simulation = Simulation(
        MonolithicConfig(
            replica_scheduler_kind=kind,
            dummy_execution_time_ms=10.0,
            build_dir=ROOT / "build",
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


def test_single_request_vllm_v1_smoke():
    simulation = _run_single_request(ReplicaSchedulerKind.VLLM_V1)
    assert simulation.all_requests_completed()
    assert simulation.completed_batches >= 1
    assert simulation.sim.now() > 0.0


def test_single_request_sglang_smoke():
    simulation = _run_single_request(ReplicaSchedulerKind.SGLANG)
    assert simulation.all_requests_completed()
    assert simulation.completed_batches >= 1


def test_multi_request_batching():
    simulation = Simulation(
        MonolithicConfig(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=10.0,
            build_dir=ROOT / "build",
        )
    )
    requests = [
        simulation.add_request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=2),
        simulation.add_request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=2),
    ]
    simulation.inject_requests(requests)
    simulation.run()
    simulation.check_errors()

    assert all(request.completed for request in requests)
    assert simulation.completed_batches >= 2


def test_time_consistency():
    simulation = Simulation(
        MonolithicConfig(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=25.0,
            build_dir=ROOT / "build",
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
    assert abs(actual - predicted) < 1e-6


def test_vllm_v1_vs_sglang_both_complete():
    vllm = _run_single_request(ReplicaSchedulerKind.VLLM_V1)
    sglang = _run_single_request(ReplicaSchedulerKind.SGLANG)
    assert vllm.all_requests_completed()
    assert sglang.all_requests_completed()


def main():
    test_single_request_vllm_v1_smoke()
    test_single_request_sglang_smoke()
    test_multi_request_batching()
    test_time_consistency()
    test_vllm_v1_vs_sglang_both_complete()
    print("All scheduler integration tests passed.")


if __name__ == "__main__":
    main()
