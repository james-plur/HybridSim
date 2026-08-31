"""Frozen memory-bound rf_op costs (layernorm / residual)."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)


def _norm_cost(model: ModelConfig, batch: BatchFeatures) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    elems = s * h
    return {"flops": 8.0 * elems, "bytes": dtype * elems * 3}


def _residual_cost(model: ModelConfig, batch: BatchFeatures) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    elems = s * h
    return {"flops": float(elems), "bytes": dtype * elems * 3}


def memory_cost_for_op(
    *,
    op_name: str,
    model: ModelConfig,
    batch: BatchFeatures,
) -> dict[str, Any]:
    if op_name in ("input_layernorm", "post_attention_layernorm"):
        return _norm_cost(model, batch)
    return _residual_cost(model, batch)
