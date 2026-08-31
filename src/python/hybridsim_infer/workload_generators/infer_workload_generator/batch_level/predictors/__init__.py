"""Batch duration predictors for batch-level infer workload generators."""

from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors.base import (
    BatchDurationPredictor,
    FixedDurationPredictor,
    TokenProportionalPredictor,
    make_predictor,
)

__all__ = [
    "BatchDurationPredictor",
    "FixedDurationPredictor",
    "TokenProportionalPredictor",
    "make_predictor",
]

try:
    from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors.frontier import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
