"""Infer workload generators: batch-level predictors and op-level mock DAG."""

from hybridsim_infer.workload_generators.infer_workload_generator.base import (
    InferWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
    BatchPhase,
    extract_batch_features,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_level import (
    BatchLevelWorkloadGenerator,
    FixedDurationPredictor,
    TokenProportionalPredictor,
    make_predictor,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level import (
    OpLevelWorkloadGenerator,
    build_operator_dag,
)

__all__ = [
    "BatchFeatures",
    "BatchLevelWorkloadGenerator",
    "BatchPhase",
    "FixedDurationPredictor",
    "InferWorkloadGenerator",
    "OpLevelWorkloadGenerator",
    "TokenProportionalPredictor",
    "build_operator_dag",
    "extract_batch_features",
    "make_predictor",
]

try:
    from hybridsim_infer.workload_generators.infer_workload_generator.batch_level import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
