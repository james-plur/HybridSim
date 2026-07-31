"""Memory operators aligned with Frontier ``memory`` family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    BatchPhase,
    OperatorKind,
)


def _norm_cost(model: ModelConfig, batch: BatchFeatures) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    elems = s * h
    # RMSNorm / LayerNorm: a few FLOPs per element, memory-bound.
    return {"flops": 8.0 * elems, "bytes": dtype * elems * 3}


def _residual_cost(model: ModelConfig, batch: BatchFeatures) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    elems = s * h
    return {"flops": float(elems), "bytes": dtype * elems * 3}


@dataclass
class MemoryOperator(Operator):
    """Layernorm / residual (Frontier memory family)."""

    def __post_init__(self) -> None:
        self.kind = OperatorKind.MEMORY

    def expand_kernels(self):
        return self._single_kernel()


def make_memory_operator(
    *,
    layer_id: int,
    op_name: str,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
) -> MemoryOperator:
    if op_name in ("input_layernorm", "post_attention_layernorm"):
        cost = _norm_cost(model, batch)
    else:
        cost = _residual_cost(model, batch)
    return MemoryOperator(
        name=f"L{layer_id}.{op_name}",
        kind=OperatorKind.MEMORY,
        variant=op_name,
        phase=batch.phase,
        deps=list(deps),
        features={"rf_op": op_name, **cost},
        layer_id=layer_id,
    )
