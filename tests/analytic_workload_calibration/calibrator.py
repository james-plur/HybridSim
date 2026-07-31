"""Offline calibration helpers for analytic OpWorkloadGenerator (tests only).

Fit a ``duration_scale`` (and optionally device/network knobs) against a
reference predictor such as Frontier RF. Production code only consumes the
resulting ``AnalyticalConfig.duration_scale`` / device params — it does not
import this package.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Sequence

from hybridsim_infer.schedule_types import ScheduleBatch
from hybridsim_infer.workload_generators import OpWorkloadGenerator
from hybridsim_infer.workload_generators.analytic_model import AnalyticalConfig


def relative_error(got: float, ref: float) -> float:
    return abs(float(got) - float(ref)) / max(abs(float(ref)), 1e-30)


def fit_duration_scale(
    analytical_s: Sequence[float],
    reference_s: Sequence[float],
    *,
    min_scale: float = 1e-12,
    max_scale: float = 1e12,
) -> float:
    """Least-squares scale: ``scale * analytical ≈ reference`` (through origin)."""
    if len(analytical_s) != len(reference_s):
        raise ValueError("analytical_s and reference_s must have the same length")
    if not analytical_s:
        raise ValueError("need at least one calibration pair")
    num = 0.0
    den = 0.0
    for a, r in zip(analytical_s, reference_s):
        a = float(a)
        r = float(r)
        if a < 0 or r < 0:
            raise ValueError("durations must be non-negative")
        num += a * r
        den += a * a
    if den <= 0.0:
        raise ValueError("analytical durations are all zero; cannot fit scale")
    scale = num / den
    return max(min_scale, min(max_scale, scale))


def measure_raw_durations(
    gen: OpWorkloadGenerator,
    batches: Sequence[ScheduleBatch],
    *,
    metric: str = "critical_path",
) -> list[float]:
    """Measure analytical durations with ``duration_scale`` forced to 1.0."""
    old = gen.analyzer.duration_scale
    gen.analyzer.duration_scale = 1.0
    try:
        return [gen.predict_duration_s(b, metric=metric) for b in batches]
    finally:
        gen.analyzer.duration_scale = old


def calibrate_duration_scale(
    gen: OpWorkloadGenerator,
    batches: Sequence[ScheduleBatch],
    reference_predict: Callable[[ScheduleBatch], float],
    *,
    metric: str = "critical_path",
) -> float:
    """Fit scale vs ``reference_predict``, install on generator + config, return it."""
    if not batches:
        raise ValueError("calibrate_duration_scale requires at least one batch")
    analytical = measure_raw_durations(gen, batches, metric=metric)
    reference = [float(reference_predict(b)) for b in batches]
    scale = fit_duration_scale(analytical, reference)
    gen.analyzer.duration_scale = scale
    gen.config.duration_scale = scale
    return scale


def calibrated_config(
    base: AnalyticalConfig,
    *,
    duration_scale: float,
) -> AnalyticalConfig:
    """Return a copy of ``base`` with calibrated ``duration_scale`` for reuse."""
    return replace(base, duration_scale=float(duration_scale))


def alignment_errors(
    gen: OpWorkloadGenerator,
    batches: Sequence[ScheduleBatch],
    reference_predict: Callable[[ScheduleBatch], float],
    *,
    metric: str = "critical_path",
) -> list[float]:
    return [
        relative_error(gen.predict_duration_s(b, metric=metric), float(reference_predict(b)))
        for b in batches
    ]
