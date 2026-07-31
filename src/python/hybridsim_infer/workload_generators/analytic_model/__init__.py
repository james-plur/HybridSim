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
    "build_operator_dag",
    "critical_path_duration_s",
    "expected_layer_op_names",
    "strip_layer_prefix",
    "total_kernel_duration_s",
]
