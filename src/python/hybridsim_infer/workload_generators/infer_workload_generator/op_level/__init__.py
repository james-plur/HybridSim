"""Op-level infer workload: mock DAG construction + analytic analyzer."""

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    AnalyticAnalyzer,
    critical_path_duration_s,
    total_kernel_duration_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.generator import (
    OpLevelWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.expect import (
    expected_layer_primitives,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    build_operator_dag,
)

__all__ = [
    "AnalyticAnalyzer",
    "OpLevelWorkloadGenerator",
    "build_operator_dag",
    "critical_path_duration_s",
    "expected_layer_primitives",
    "total_kernel_duration_s",
]
