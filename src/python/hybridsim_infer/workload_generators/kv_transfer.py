"""KV cache transfer workload generator (pull/push TimeoutKernel)."""

from __future__ import annotations

from typing import Any, Literal

TransferDirection = Literal["pull", "push"]


class KvTransferWorkloadGenerator:
    """Build EngineActor workloads for remote KV pull/push transfers.

    Extension point for future transfer models (bandwidth curves, multi-kernel
    pipelines, connector metadata). Callers typically pass duration already
    estimated by ``KvClient.transfer_duration_s``.
    """

    def __call__(
        self,
        *,
        workload_id: int,
        request_id: int,
        duration_s: float,
        direction: TransferDirection = "pull",
        num_tokens: int = 0,
    ) -> dict[str, Any]:
        _ = num_tokens  # reserved for bandwidth / multi-chunk extensions
        dir_tag = str(direction or "pull")
        return {
            "workload_id": int(workload_id),
            "kernels": [
                {
                    "name": f"kv_xfer_{dir_tag}_{int(request_id)}",
                    "duration": float(duration_s),
                    "dependencies": [],
                }
            ],
        }
