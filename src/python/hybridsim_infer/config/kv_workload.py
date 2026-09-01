"""KV data-plane transfer duration (α-β). Distinct from op-level collectives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KvWorkloadConfig:
    """Interconnect model for KV pull/push TimeoutKernel duration."""

    #: Simulated interconnect bandwidth (Gbps).
    bandwidth_gbps: float = 50.0
    #: Fixed latency α (seconds).
    latency_s: float = 0.0
    #: Floor on KV transfer TimeoutKernel duration (seconds).
    transfer_s_floor: float = 1e-4
    #: Bytes/token fallback when ``model`` is unset. Prefer ``ModelSpec``.
    bytes_per_token: float | None = None
