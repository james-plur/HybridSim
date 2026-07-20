#!/usr/bin/env python3
"""Minimal MONOLITHIC scheduler demo for hybridsim + Frontier integration.

Requires Frontier installed as a third-party package:
  pip install -e /path/to/Frontier
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from hybridsim_scheduler import MonolithicConfig, ReplicaSchedulerKind, Simulation


def main() -> None:
    simulation = Simulation(
        MonolithicConfig(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=50.0,
            build_dir=ROOT / "build",
        )
    )

    simulation.inject_requests(
        [
            simulation.add_request(
                arrived_at=0.0,
                num_prefill_tokens=8,
                num_decode_tokens=4,
            ),
            simulation.add_request(
                arrived_at=0.0,
                num_prefill_tokens=16,
                num_decode_tokens=2,
            ),
        ]
    )

    simulation.run()
    simulation.check_errors()

    print(f"simulation finished at t={simulation.sim.now():.6f}s")
    print(f"completed_batches={simulation.completed_batches}")
    print(f"predicted_duration_total={simulation.predicted_duration_total:.6f}s")
    print(
        "requests_completed="
        f"{sum(1 for request in simulation.requests if request.completed)}"
        f"/{len(simulation.requests)}"
    )


if __name__ == "__main__":
    main()
