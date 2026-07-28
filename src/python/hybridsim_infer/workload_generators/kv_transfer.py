"""KV transfer workloads (pull/push TimeoutKernel)."""

from __future__ import annotations

from typing import Any


def kv_transfer_workload(
    *,
    workload_id: int,
    request_id: int,
    duration_s: float,
) -> dict[str, Any]:
    """Build an EngineActor workload for one KV pull/push transfer."""
    return {
        "workload_id": int(workload_id),
        "kernels": [
            {
                "name": f"kv_xfer_{int(request_id)}",
                "duration": float(duration_s),
                "dependencies": [],
            }
        ],
    }
