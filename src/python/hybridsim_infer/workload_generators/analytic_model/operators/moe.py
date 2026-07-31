"""MoE operators aligned with Frontier ``moe`` / ``share_expert`` families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
    MOE_OPS,
    SHARE_EXPERT_OPS,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    OperatorKind,
)


def _moe_costs(
    *,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int,
) -> dict[str, dict[str, Any]]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    i = max(1, int(model.intermediate_size))
    e = max(1, int(model.num_experts))
    k = max(1, int(model.num_experts_per_tok))
    tp = max(1, int(tp_size))
    dtype = max(1, int(model.dtype_bytes))
    i_local = max(1, i // tp)

    # Gating: hidden → num_experts
    gate = {
        "flops": 2.0 * s * h * e,
        "bytes": dtype * (s * h + h * e + s * e),
    }
    # TopK + softmax over experts
    topk = {
        "flops": 8.0 * s * e,
        "bytes": dtype * s * e * 2,
    }
    # Token shuffle / permute overhead (approximate bandwidth-bound)
    shuffle = {
        "flops": 0.0,
        "bytes": dtype * s * h * 2,
    }
    # Grouped GEMM: active tokens * k experts * (up+down), SwiGLU-style
    active = s * k
    gemm = {
        "flops": 2.0 * active * h * i_local * 2  # up (gated) + down
        + 2.0 * active * i_local * h,
        "bytes": dtype
        * (
            active * h
            + 2 * h * i_local  # weights approx per expert shard (shared)
            + active * i_local
            + i_local * h
            + active * h
        ),
    }
    return {
        "moe_gating_linear": gate,
        "moe_gating_routing_topk": topk,
        "moe_shuffling": shuffle,
        "moe_grouped_gemm": gemm,
    }


def _share_expert_costs(
    *,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int,
) -> dict[str, dict[str, Any]]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    i = max(1, int(model.share_expert_dim))
    tp = max(1, int(tp_size))
    dtype = max(1, int(model.dtype_bytes))
    i_local = max(1, i // tp)
    return {
        "share_expert_up_proj": {
            "flops": 2.0 * s * h * i_local * 2,
            "bytes": dtype * (s * h + 2 * h * i_local + 2 * s * i_local),
        },
        "share_expert_act": {
            "flops": 8.0 * s * i_local * 2,
            "bytes": dtype * s * i_local * 4,
        },
        "share_expert_down_proj": {
            "flops": 2.0 * s * i_local * h,
            "bytes": dtype * (s * i_local + i_local * h + s * h),
        },
    }


@dataclass
class MoEOperator(Operator):
    """Frontier MoE / share_expert physical operator."""

    def __post_init__(self) -> None:
        self.kind = OperatorKind.COMPUTE_MOE

    def expand_kernels(self):
        return self._single_kernel()


def make_moe_block_operators(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
) -> list[MoEOperator]:
    costs = _moe_costs(model=model, batch=batch, tp_size=tp_size)
    ops: list[MoEOperator] = []
    for op_name in MOE_OPS:
        ops.append(
            MoEOperator(
                name=f"L{layer_id}.{op_name}",
                kind=OperatorKind.COMPUTE_MOE,
                variant="moe",
                phase=batch.phase,
                deps=[],
                features={"rf_op": op_name, **costs[op_name]},
                layer_id=layer_id,
            )
        )
    if ops:
        ops[0].deps = list(deps)
    return ops


def make_share_expert_operators(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
) -> list[MoEOperator]:
    if not model.has_share_expert():
        return []
    costs = _share_expert_costs(model=model, batch=batch, tp_size=tp_size)
    ops: list[MoEOperator] = []
    for op_name in SHARE_EXPERT_OPS:
        ops.append(
            MoEOperator(
                name=f"L{layer_id}.{op_name}",
                kind=OperatorKind.COMPUTE_MOE,
                variant="share_expert",
                phase=batch.phase,
                deps=[],
                features={"rf_op": op_name, **costs[op_name]},
                layer_id=layer_id,
            )
        )
    if ops:
        ops[0].deps = list(deps)
    return ops
