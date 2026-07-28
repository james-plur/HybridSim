"""Native Actor-based inference simulation (NO_NETWORK skeleton)."""

from hybridsim_infer.builder import InferenceSimulation, build_inference_simulation
from hybridsim_infer.config import InferenceConfig
from hybridsim_infer.frameworks import FrameworkFactory, InferenceFramework, VllmFramework
from hybridsim_infer.messages import INFER_MESSAGE_TYPES
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.request_generators import (
    ListRequestGenerator,
    RequestGenerator,
    ServeGenRequestGenerator,
    map_servegen_request,
)
from hybridsim_infer.workload_generators import (
    FixedDurationPredictor,
    PredictWorkloadGenerator,
    TokenProportionalPredictor,
    WorkloadGenerator,
    make_predictor,
    make_workload_generator,
)

__all__ = [
    "FrameworkFactory",
    "INFER_MESSAGE_TYPES",
    "FixedDurationPredictor",
    "InferenceConfig",
    "InferenceFramework",
    "InferenceRequest",
    "InferenceSimulation",
    "ListRequestGenerator",
    "PredictWorkloadGenerator",
    "RequestGenerator",
    "RequestStatus",
    "ServeGenRequestGenerator",
    "TokenProportionalPredictor",
    "VllmFramework",
    "WorkloadGenerator",
    "build_inference_simulation",
    "make_predictor",
    "make_workload_generator",
    "map_servegen_request",
]

try:
    from hybridsim_infer.workload_generators import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
