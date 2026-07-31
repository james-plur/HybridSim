"""Roofline time model: T = max(flops/peak, bytes/bandwidth)."""

from __future__ import annotations

from hybridsim_infer.workload_generators.analytic_model.configs import DeviceConfig


def roofline_time_s(
    *,
    flops: float,
    bytes_: float,
    device: DeviceConfig,
) -> float:
    """Estimate kernel duration in seconds from arithmetic intensity."""
    peak = max(1e-30, float(device.peak_flops))
    bw = max(1e-30, float(device.hbm_bandwidth_bps))
    t_compute = max(0.0, float(flops)) / peak
    t_memory = max(0.0, float(bytes_)) / bw
    return max(t_compute, t_memory)
