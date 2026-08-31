"""Frozen FFN rf_op costs."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.types import FfnActivation


def ffn_component_costs(
    *,
    activation: FfnActivation,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, dict[str, Any]]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    i = max(1, int(model.intermediate_size))
    tp = max(1, int(tp_size))
    dtype = max(1, int(model.dtype_bytes))
    i_local = max(1, i // tp)
    gated = activation in (FfnActivation.SILU, FfnActivation.SWIGLU)
    up_mats = 2 if gated else 1

    flops_up = 2.0 * s * h * i_local * up_mats
    bytes_up = dtype * (s * h + up_mats * h * i_local + up_mats * s * i_local)

    act_elems = s * i_local * (2 if gated else 1)
    flops_act = 8.0 * act_elems
    bytes_act = dtype * act_elems * 2

    flops_down = 2.0 * s * i_local * h
    bytes_down = dtype * (s * i_local + i_local * h + s * h)

    return {
        "mlp_up_proj": {"flops": flops_up, "bytes": bytes_up},
        "mlp_act": {"flops": flops_act, "bytes": bytes_act},
        "mlp_down_proj": {"flops": flops_down, "bytes": bytes_down},
    }


def ffn_cost_for_op(
    *,
    op_name: str,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, Any]:
    parts = ffn_component_costs(
        activation=model.resolved_ffn_activation(),
        model=model,
        batch=batch,
        tp_size=tp_size,
    )
    return parts.get(op_name, {"flops": 0.0, "bytes": 0.0})
