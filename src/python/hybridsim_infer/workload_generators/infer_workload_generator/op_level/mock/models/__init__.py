"""Mock transformer modules used to expand an Operator DAG via ``forward``."""

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.attention import (
    Attention,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.ffn import (
    FFN,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.moe import (
    MoE,
    ShareExpert,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.expect import (
    expected_layer_primitives,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    Transformer,
    build_operator_dag,
)

__all__ = [
    "Attention",
    "FFN",
    "MoE",
    "ShareExpert",
    "Transformer",
    "build_operator_dag",
    "expected_layer_primitives",
]
