"""Batch duration predictors for workload generators.

Generators live in the parent package; all predictor logic lives here.
"""

from hybridsim_infer.workload_generators.predictors.base import (
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
    from hybridsim_infer.workload_generators.predictors.frontier import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
