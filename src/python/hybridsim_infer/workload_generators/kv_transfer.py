"""KV cache transfer workload generator (α-β TimeoutKernel)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from hybridsim_infer.workload_generators.analytic_model.configs import (
    ModelConfig,
    NetworkConfig,
)
from hybridsim_infer.workload_generators.analytic_model.kv_cache import (
    bytes_per_token,
    cache_bytes,
    resolve_model,
)

TransferDirection = Literal["pull", "push"]


def transfer_duration_s(
    *,
    num_tokens: int,
    model: Optional[ModelConfig | str] = None,
    bytes_per_token_fallback: float = 16.0,
    network: Optional[NetworkConfig] = None,
    bandwidth_gbps: float = 50.0,
    latency_s: float = 0.0,
    transfer_s_floor: float = 0.0,
) -> float:
    """α-β transfer time: ``latency + bytes / bandwidth``.

    When ``model`` is a ``ModelConfig`` or preset id, bytes come from
    :func:`cache_bytes`; otherwise ``num_tokens * bytes_per_token_fallback``.
    """
    tokens = max(0, int(num_tokens))
    if model is not None:
        model = resolve_model(model)
        nbytes = cache_bytes(model, tokens)
    else:
        nbytes = float(tokens) * float(bytes_per_token_fallback)

    if network is not None:
        alpha = float(network.alpha_s)
        beta = float(network.beta_s_per_byte)
        duration = alpha + nbytes * beta
    else:
        bps = max(1e-9, float(bandwidth_gbps)) * (1e9 / 8.0)
        duration = float(latency_s) + nbytes / bps
    return max(float(transfer_s_floor), float(duration))


class KvTransferWorkloadGenerator:
    """Build EngineActor workloads for remote KV pull/push transfers.

    Duration uses model-driven KV volume + α-β interconnect when configured;
    callers may still pass a precomputed ``duration_s``.
    """

    def __init__(
        self,
        *,
        model: Optional[ModelConfig | str] = None,
        network: Optional[NetworkConfig] = None,
        bytes_per_token: float = 16.0,
        bandwidth_gbps: float = 50.0,
        latency_s: float = 0.0,
        transfer_s_floor: float = 0.0,
        page_tokens: int = 0,
    ) -> None:
        self.model = resolve_model(model) if model is not None else None
        self.network = network
        self.bytes_per_token = float(bytes_per_token)
        self.bandwidth_gbps = float(bandwidth_gbps)
        self.latency_s = float(latency_s)
        self.transfer_s_floor = float(transfer_s_floor)
        #: When >0, split transfer into serial TimeoutKernels of this many tokens.
        self.page_tokens = max(0, int(page_tokens))

    def estimate_duration_s(self, num_tokens: int) -> float:
        return transfer_duration_s(
            num_tokens=num_tokens,
            model=self.model,
            bytes_per_token_fallback=self.bytes_per_token,
            network=self.network,
            bandwidth_gbps=self.bandwidth_gbps,
            latency_s=self.latency_s,
            transfer_s_floor=self.transfer_s_floor,
        )

    def estimate_bytes(self, num_tokens: int) -> float:
        tokens = max(0, int(num_tokens))
        if self.model is not None:
            return cache_bytes(self.model, tokens)
        return float(tokens) * self.bytes_per_token

    def __call__(
        self,
        *,
        workload_id: int,
        request_id: int,
        duration_s: float | None = None,
        direction: TransferDirection = "pull",
        num_tokens: int = 0,
    ) -> dict[str, Any]:
        dir_tag = str(direction or "pull")
        tokens = max(0, int(num_tokens))
        page = self.page_tokens

        if page > 0 and tokens > page:
            kernels: list[dict[str, Any]] = []
            remaining = tokens
            idx = 0
            prev_name: str | None = None
            while remaining > 0:
                chunk = min(page, remaining)
                chunk_dur = self.estimate_duration_s(chunk)
                # α only once for the whole transfer; subsequent pages are β-only.
                if idx > 0 and self.network is not None:
                    nbytes = self.estimate_bytes(chunk)
                    chunk_dur = max(
                        self.transfer_s_floor,
                        nbytes * float(self.network.beta_s_per_byte),
                    )
                elif idx > 0:
                    nbytes = self.estimate_bytes(chunk)
                    bps = max(1e-9, self.bandwidth_gbps) * (1e9 / 8.0)
                    chunk_dur = max(self.transfer_s_floor, nbytes / bps)
                name = f"kv_xfer_{dir_tag}_{int(request_id)}_p{idx}"
                deps = [prev_name] if prev_name is not None else []
                # TimeoutKernel deps are indices into kernels list when ints;
                # hybridsim engine uses dependency names or indices — match
                # existing pattern of empty deps + serial order via index deps.
                kernels.append(
                    {
                        "name": name,
                        "duration": float(chunk_dur),
                        "dependencies": [idx - 1] if idx > 0 else [],
                    }
                )
                prev_name = name
                remaining -= chunk
                idx += 1
            return {"workload_id": int(workload_id), "kernels": kernels}

        dur = (
            float(duration_s)
            if duration_s is not None
            else self.estimate_duration_s(tokens)
        )
        return {
            "workload_id": int(workload_id),
            "kernels": [
                {
                    "name": f"kv_xfer_{dir_tag}_{int(request_id)}",
                    "duration": float(dur),
                    "dependencies": [],
                }
            ],
        }


# Re-export helpers used by KvClient / tests.
__all__ = [
    "KvTransferWorkloadGenerator",
    "TransferDirection",
    "bytes_per_token",
    "cache_bytes",
    "transfer_duration_s",
]
