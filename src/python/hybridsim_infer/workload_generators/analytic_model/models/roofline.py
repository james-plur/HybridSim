"""Roofline time model: T = max(flops/peak_eff, bytes/bandwidth_eff)."""

from __future__ import annotations

from hybridsim_infer.workload_generators.analytic_model.configs import DeviceConfig


def roofline_time_s(
    *,
    flops: float,
    bytes_: float,
    device: DeviceConfig,
) -> float:
    """Estimate kernel duration in seconds from arithmetic intensity.

    Uses effective peaks: ``peak_flops * compute_util`` and
    ``hbm_bandwidth_bps * hbm_util`` (datasheet peaks are rarely fully achieved).
    """
    peak = device.effective_peak_flops()
    bw = device.effective_hbm_bandwidth_bps()
    t_compute = max(0.0, float(flops)) / peak
    t_memory = max(0.0, float(bytes_)) / bw
    return max(t_compute, t_memory)
