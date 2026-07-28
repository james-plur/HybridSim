"""Factories for workload generators."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.workload_generators.base import WorkloadGenerator
from hybridsim_infer.workload_generators.predictors import (
    BatchDurationPredictor,
    make_predictor,
)
from hybridsim_infer.workload_generators.timeout_kernel import (
    TimeoutKernelWorkloadGenerator,
)


def make_workload_generator(
    *,
    duration_mode: str = "fixed",
    dummy_exec_s: float = 0.05,
    prefill_s_per_token: float = 1e-4,
    decode_s_per_token: float = 1e-3,
    duration_base_s: float = 0.0,
    predictor: Optional[BatchDurationPredictor] = None,
) -> WorkloadGenerator:
    """Default: ``TimeoutKernelWorkloadGenerator`` over a duration predictor."""
    pred = predictor or make_predictor(
        duration_mode=duration_mode,
        dummy_exec_s=dummy_exec_s,
        prefill_s_per_token=prefill_s_per_token,
        decode_s_per_token=decode_s_per_token,
        base_s=duration_base_s,
    )
    return TimeoutKernelWorkloadGenerator(pred)


def make_workload_generator_from_config(config: Any) -> WorkloadGenerator:
    return make_workload_generator(
        duration_mode=getattr(config, "duration_mode", "fixed"),
        dummy_exec_s=getattr(config, "dummy_exec_s", 0.05),
        prefill_s_per_token=getattr(config, "prefill_s_per_token", 1e-4),
        decode_s_per_token=getattr(config, "decode_s_per_token", 1e-3),
        duration_base_s=getattr(config, "duration_base_s", 0.0),
    )
