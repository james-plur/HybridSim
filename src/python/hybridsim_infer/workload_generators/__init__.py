"""Pluggable workload generators (ScheduleBatch → EngineActor workload).

Sibling of ``frameworks/``, ``kv_system/``, and ``actors/``.
"""

from hybridsim_infer.workload_generators.base import WorkloadGenerator
from hybridsim_infer.workload_generators.factory import (
    make_workload_generator,
    make_workload_generator_from_config,
)
from hybridsim_infer.workload_generators.kv_transfer import kv_transfer_workload
from hybridsim_infer.workload_generators.predictors import (
    BatchDurationPredictor,
    FixedDurationPredictor,
    TokenProportionalPredictor,
    make_predictor,
)
from hybridsim_infer.workload_generators.timeout_kernel import (
    TimeoutKernelWorkloadGenerator,
)

__all__ = [
    "BatchDurationPredictor",
    "FixedDurationPredictor",
    "TimeoutKernelWorkloadGenerator",
    "TokenProportionalPredictor",
    "WorkloadGenerator",
    "kv_transfer_workload",
    "make_predictor",
    "make_workload_generator",
    "make_workload_generator_from_config",
]
