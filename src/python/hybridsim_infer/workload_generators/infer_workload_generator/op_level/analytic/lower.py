"""Lower a shape-primitive Op into a KernelPlan (flops/bytes or comm features)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    KernelPlan,
    OperatorKind,
)

if TYPE_CHECKING:
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
        Op,
    )


def lower_op(op: Op) -> KernelPlan:
    """Attach Roofline / α-β work features from the op itself."""
    kind = op.kind
    return KernelPlan(
        name=op.name,
        features=op.features(),
        kind=kind if isinstance(kind, OperatorKind) else OperatorKind.GEMM,
    )
