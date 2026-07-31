"""FFN operators aligned with Frontier ``ffn`` family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import FFN_OPS
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    FfnActivation,
    OperatorKind,
)


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


def ffn_compute_cost(
    *,
    activation: FfnActivation,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, Any]:
    parts = ffn_component_costs(
        activation=activation, model=model, batch=batch, tp_size=tp_size
    )
    return {
        "flops": sum(p["flops"] for p in parts.values()),
        "bytes": sum(p["bytes"] for p in parts.values()),
        "flops_up": parts["mlp_up_proj"]["flops"],
        "bytes_up": parts["mlp_up_proj"]["bytes"],
        "flops_act": parts["mlp_act"]["flops"],
        "bytes_act": parts["mlp_act"]["bytes"],
        "flops_down": parts["mlp_down_proj"]["flops"],
        "bytes_down": parts["mlp_down_proj"]["bytes"],
    }


@dataclass
class FfnOperator(Operator):
    """Single Frontier FFN physical op (``mlp_up_proj`` / ``mlp_act`` / ``mlp_down_proj``)."""

    def __post_init__(self) -> None:
        self.kind = OperatorKind.COMPUTE_FFN

    def expand_kernels(self):
        return self._single_kernel()


def make_ffn_block_operators(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
) -> list[FfnOperator]:
    activation = model.resolved_ffn_activation()
    parts = ffn_component_costs(
        activation=activation, model=model, batch=batch, tp_size=tp_size
    )
    ops: list[FfnOperator] = []
    for op_name in FFN_OPS:
        cost = parts[op_name]
        ops.append(
            FfnOperator(
                name=f"L{layer_id}.{op_name}",
                kind=OperatorKind.COMPUTE_FFN,
                variant=activation.value,
                phase=batch.phase,
                deps=[],
                features={
                    "rf_op": op_name,
                    "num_tokens": batch.num_tokens,
                    **cost,
                },
                layer_id=layer_id,
            )
        )
    if ops:
        ops[0].deps = list(deps)
    return ops


def make_ffn_operator(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
    expand_sub_kernels: bool = True,
) -> FfnOperator:
    """Legacy single-node FFN; prefer ``make_ffn_block_operators``."""
    del expand_sub_kernels
    cost = ffn_compute_cost(
        activation=model.resolved_ffn_activation(),
        model=model,
        batch=batch,
        tp_size=tp_size,
    )
    return FfnOperator(
        name=f"L{layer_id}.ffn",
        kind=OperatorKind.COMPUTE_FFN,
        variant=model.resolved_ffn_activation().value,
        phase=batch.phase,
        deps=list(deps),
        features={"rf_op": "ffn_fused", **cost},
        layer_id=layer_id,
    )
