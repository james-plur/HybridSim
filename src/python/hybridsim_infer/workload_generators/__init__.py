"""Pluggable workload generators (ScheduleBatch / KV transfer → EngineActor).

Sibling of ``schedulers/``, ``kv_system/``, and ``actors/``.

Layout:
  - infer_workload_generator: batch_level predictors + op_level mock DAG
  - kv_workload_generator: α-β KV pull/push
  - model_presets / configs / kv_cache: shared by both
"""

from hybridsim_infer.workload_generators.configs import (
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    OpLevelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.factory import (
    make_infer_workload_generator,
    make_kv_workload_generator,
)
from hybridsim_infer.workload_generators.infer_workload_generator import (
    BatchLevelWorkloadGenerator,
    InferWorkloadGenerator,
    OpLevelWorkloadGenerator,
    extract_batch_features,
    make_predictor,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_level import (
    BatchDurationPredictor,
    FixedDurationPredictor,
    TokenProportionalPredictor,
)
from hybridsim_infer.workload_generators.kv_cache import bytes_per_token, cache_bytes
from hybridsim_infer.workload_generators.kv_workload_generator import (
    KvWorkloadGenerator,
)
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_model_config,
    resolve_op_level_config,
)
from hybridsim_infer.workload_generators.model_presets import (
    list_presets,
    load_model_config,
    load_preset,
)
from hybridsim_infer.workload_generators.types import (
    AttnVariant,
    FfnActivation,
    ensure_attn_variant_supported,
)

__all__ = [
    "AttnVariant",
    "BatchDurationPredictor",
    "BatchLevelWorkloadGenerator",
    "DeviceConfig",
    "FfnActivation",
    "FixedDurationPredictor",
    "InferWorkloadGenerator",
    "KvWorkloadGenerator",
    "ModelConfig",
    "NetworkConfig",
    "OpLevelConfig",
    "OpLevelWorkloadGenerator",
    "ParallelConfig",
    "TokenProportionalPredictor",
    "bytes_per_token",
    "ensure_attn_variant_supported",
    "cache_bytes",
    "extract_batch_features",
    "list_presets",
    "load_model_config",
    "load_preset",
    "make_infer_workload_generator",
    "make_kv_workload_generator",
    "make_predictor",
    "resolve_model_config",
    "resolve_op_level_config",
]

try:
    from hybridsim_infer.workload_generators.infer_workload_generator import (  # noqa: F401
        FrontierBatchDurationPredictor,
    )

    __all__.append("FrontierBatchDurationPredictor")
except ImportError:
    pass
