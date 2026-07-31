"""α-β communication time model."""

from __future__ import annotations

from hybridsim_infer.workload_generators.analytic_model.configs import NetworkConfig


def ab_comm_time_s(
    *,
    payload_bytes: float,
    volume_factor: float,
    network: NetworkConfig,
    num_ranks: int = 1,
) -> float:
    """T = alpha + beta * payload_bytes * volume_factor.

    When ``num_ranks <= 1`` or ``volume_factor == 0``, returns 0 (no communication).
    """
    if int(num_ranks) <= 1 or float(volume_factor) <= 0.0:
        return 0.0
    moved = max(0.0, float(payload_bytes)) * max(0.0, float(volume_factor))
    return float(network.alpha_s) + float(network.beta_s_per_byte) * moved
