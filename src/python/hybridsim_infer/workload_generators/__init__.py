"""Pluggable workload generators (ScheduleBatch → EngineActor workload).

Sibling of ``frameworks/``, ``kv_system/``, and ``actors/``.

Layout:
  - generators: ``base``, ``predict``, ``op_workload_generator``, ``kv_transfer``
  - analytic_model: Operator DAG + Roofline / α-β OpAnalyzer
  - predictors: ``predictors/`` (fixed / token-proportional / Frontier)
"""

from hybridsim_infer.workload_generators.base import WorkloadGenerator
from hybridsim_infer.workload_generators.factory import (
    make_workload_generator,
)
from hybridsim_infer.workload_generators.kv_transfer import (
    KvTransferWorkloadGenerator,
)
from hybridsim_infer.workload_generators.op_workload_generator import (
    OpWorkloadGenerator,
    extract_batch_features,
)
from hybridsim_infer.workload_generators.predict import PredictWorkloadGenerator
from hybridsim_infer.workload_generators.predictors import (
    BatchDurationPredictor,
    FixedDurationPredictor,
    TokenProportionalPredictor,
    make_predictor,
)

__all__ = [
    "BatchDurationPredictor",
    "FixedDurationPredictor",
    "KvTransferWorkloadGenerator",
    "OpWorkloadGenerator",
    "PredictWorkloadGenerator",
    "TokenProportionalPredictor",
    "WorkloadGenerator",
    "extract_batch_features",
    "make_predictor",
    "make_workload_generator",
]

try:
    from hybridsim_infer.workload_generators.predictors import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
