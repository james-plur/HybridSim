"""Analytic Operator model: DAG construction, Roofline / α-β, OpAnalyzer."""

from hybridsim_infer.workload_generators.analytic_model.configs import (
    AnalyticalConfig,
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.analytic_model.dag_builder import (
    build_operator_dag,
)
from hybridsim_infer.workload_generators.analytic_model.dag_profile import (
    asap_schedule,
    build_chrome_trace,
    profile_schedule_batch,
    write_chrome_trace,
)
from hybridsim_infer.workload_generators.analytic_model.kv_cache import (
    bytes_per_token,
    cache_bytes,
)
from hybridsim_infer.workload_generators.analytic_model.model_presets import (
    list_presets,
    load_model_config,
    load_preset,
)
from hybridsim_infer.workload_generators.analytic_model.op_analyzer import (
    OpAnalyzer,
    critical_path_duration_s,
    total_kernel_duration_s,
)
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
    expected_layer_op_names,
    strip_layer_prefix,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    AttnVariant,
    BatchFeatures,
    BatchPhase,
    CommCollective,
    FfnActivation,
    OperatorDAG,
    OperatorKind,
    TpCommStyle,
)

__all__ = [
    "AnalyticalConfig",
    "AttnVariant",
    "BatchFeatures",
    "BatchPhase",
    "CommCollective",
    "DeviceConfig",
    "FfnActivation",
    "ModelConfig",
    "NetworkConfig",
    "OpAnalyzer",
    "OperatorDAG",
    "OperatorKind",
    "ParallelConfig",
    "TpCommStyle",
    "asap_schedule",
    "build_chrome_trace",
    "bytes_per_token",
    "build_operator_dag",
    "cache_bytes",
    "critical_path_duration_s",
    "expected_layer_op_names",
    "list_presets",
    "load_model_config",
    "load_preset",
    "profile_schedule_batch",
    "strip_layer_prefix",
    "total_kernel_duration_s",
    "write_chrome_trace",
]
