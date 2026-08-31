"""Batch-level infer workload: predict duration from ScheduleBatch features."""

from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.generator import (
    BatchLevelWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors import (
    BatchDurationPredictor,
    FixedDurationPredictor,
    TokenProportionalPredictor,
    make_predictor,
)

__all__ = [
    "BatchDurationPredictor",
    "BatchLevelWorkloadGenerator",
    "FixedDurationPredictor",
    "TokenProportionalPredictor",
    "make_predictor",
]

try:
    from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
