"""Frozen MoE / share-expert rf_op costs."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)


def moe_component_costs(
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

    gate = {
        "flops": 2.0 * s * h * e,
        "bytes": dtype * (s * h + h * e + s * e),
    }
    topk = {
        "flops": 8.0 * s * e,
        "bytes": dtype * s * e * 2,
    }
    shuffle = {
        "flops": 0.0,
        "bytes": dtype * s * h * 2,
    }
    active = s * k
    gemm = {
        "flops": 2.0 * active * h * i_local * 2 + 2.0 * active * i_local * h,
        "bytes": dtype
        * (
            active * h
            + 2 * h * i_local
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


def share_expert_component_costs(
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


def moe_cost_for_op(
    *,
    op_name: str,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, Any]:
    if op_name.startswith("share_expert"):
        parts = share_expert_component_costs(
            model=model, batch=batch, tp_size=tp_size
        )
        return parts.get(op_name, {"flops": 0.0, "bytes": 0.0})
    parts = moe_component_costs(model=model, batch=batch, tp_size=tp_size)
    return parts.get(op_name, {"flops": 0.0, "bytes": 0.0})
