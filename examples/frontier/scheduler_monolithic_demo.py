#!/usr/bin/env python3
"""Minimal MONOLITHIC scheduler demo for hybridsim + Frontier.

Requires:
  pip install -e /path/to/hybridsim
  pip install -e /path/to/Frontier
  PYTHONPATH=examples/frontier  (or run from this directory)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontier_bridge import (
    MonolithicConfig,
    ReplicaSchedulerKind,
    build_frontier_simulation,
)


def main() -> None:
    simulation = build_frontier_simulation(
        MonolithicConfig(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=50.0,
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
