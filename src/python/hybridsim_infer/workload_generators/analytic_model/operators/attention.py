"""Attention operators aligned with Frontier dense / MLA families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
    attn_ops_for_variant,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    AttnVariant,
    BatchFeatures,
    BatchPhase,
    OperatorKind,
)

_IMPLEMENTED = frozenset(
    {AttnVariant.MHA, AttnVariant.GQA, AttnVariant.MQA, AttnVariant.MLA}
)
_STUBBED = frozenset({AttnVariant.CSA, AttnVariant.HSA, AttnVariant.DSA})


def ensure_attn_variant_supported(variant: AttnVariant) -> None:
    if variant in _STUBBED:
        raise NotImplementedError(
            f"Attention variant {variant.value!r} is registered as an extension "
            "point but not implemented yet (supported: mha, gqa, mqa, mla)"
        )
    if variant not in _IMPLEMENTED:
        raise ValueError(f"Unknown attention variant: {variant!r}")


def _dense_head_counts(
    variant: AttnVariant, model: ModelConfig, tp: int
) -> tuple[int, int, int]:
    n_q = max(1, int(model.num_q_heads))
    n_kv = max(1, int(model.num_kv_heads))
    if variant is AttnVariant.MHA:
        n_kv = n_q
    elif variant is AttnVariant.MQA:
        n_kv = 1
    d = max(1, int(model.get_head_dim()))
    return max(1, n_q // tp), max(1, n_kv // tp), d


def _cost_pre_proj(
    *, variant: AttnVariant, model: ModelConfig, batch: BatchFeatures, tp: int
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    n_q_l, n_kv_l, d = _dense_head_counts(variant, model, tp)
    flops = 2.0 * s * h * (n_q_l * d)  # Q
    flops += 2.0 * s * h * (n_kv_l * d) * 2.0  # K, V
    nbytes = dtype * (
        s * h + h * (n_q_l * d) + h * (n_kv_l * d) * 2 + s * (n_q_l + 2 * n_kv_l) * d
    )
    return {"flops": flops, "bytes": nbytes}


def _cost_post_proj(
    *, variant: AttnVariant, model: ModelConfig, batch: BatchFeatures, tp: int
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    n_q_l, _, d = _dense_head_counts(variant, model, tp)
    flops = 2.0 * s * (n_q_l * d) * h
    nbytes = dtype * (s * n_q_l * d + (n_q_l * d) * h + s * h)
    return {"flops": flops, "bytes": nbytes}


def _cost_rope(
    *, variant: AttnVariant, model: ModelConfig, batch: BatchFeatures, tp: int
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    dtype = max(1, int(model.dtype_bytes))
    n_q_l, n_kv_l, d = _dense_head_counts(variant, model, tp)
    elems = s * (n_q_l + n_kv_l) * d
    return {"flops": 8.0 * elems, "bytes": dtype * elems * 2}


def _cost_kv_cache_save(
    *, variant: AttnVariant, model: ModelConfig, batch: BatchFeatures, tp: int
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    dtype = max(1, int(model.dtype_bytes))
    _, n_kv_l, d = _dense_head_counts(variant, model, tp)
    elems = s * n_kv_l * d * 2
    return {"flops": 0.0, "bytes": dtype * elems}


def _cost_attn_core(
    *,
    variant: AttnVariant,
    model: ModelConfig,
    batch: BatchFeatures,
    tp: int,
    which: str,
) -> dict[str, Any]:
    dtype = max(1, int(model.dtype_bytes))
    n_q_l, n_kv_l, d = _dense_head_counts(variant, model, tp)
    if which == "attn_decode":
        s = max(0, int(batch.num_decode_tokens) or (1 if batch.phase is BatchPhase.DECODE else 0))
        mean_ctx = max(
            1.0,
            batch.kv_cache_tokens / max(1, batch.num_decode_tokens or batch.batch_size),
        )
        flops = 4.0 * s * n_q_l * d * mean_ctx
        nbytes = dtype * (
            s * n_q_l * d + s * mean_ctx * n_kv_l * d * 2 + s * n_q_l * d
        )
        return {"flops": flops, "bytes": nbytes}

    # prefill
    s = max(0, int(batch.num_prefill_tokens) or int(batch.num_tokens))
    flops = 4.0 * n_q_l * d * s * s
    nbytes = dtype * (s * n_q_l * d + s * n_kv_l * d * 2 + s * n_q_l * d)
    return {"flops": flops, "bytes": nbytes}


def _mla_cost_for_op(
    *,
    op: str,
    model: ModelConfig,
    batch: BatchFeatures,
    tp: int,
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    h = max(1, int(model.hidden_size))
    dtype = max(1, int(model.dtype_bytes))
    kv_rank = max(1, int(model.kv_lora_rank))
    rope_d = max(1, int(model.qk_rope_head_dim))
    nope_d = max(1, int(model.qk_nope_head_dim))
    v_d = max(1, int(model.v_head_dim))
    n_q_l = max(1, int(model.num_q_heads) // max(1, tp))
    latent = kv_rank + rope_d

    if op == "attn_mla_kv_cache_save":
        elems = s * latent
        return {"flops": 0.0, "bytes": dtype * elems}
    if op == "attn_mla_prefill_kv_up_proj":
        flops = 2.0 * s * kv_rank * n_q_l * (nope_d + v_d)
        return {"flops": flops, "bytes": dtype * (s * kv_rank + s * n_q_l * (nope_d + v_d))}
    if op == "attn_mla_decode_q_latent_proj":
        q_rank = int(model.q_lora_rank) or h
        flops = 2.0 * s * h * q_rank + 2.0 * s * q_rank * n_q_l * (nope_d + rope_d)
        return {"flops": flops, "bytes": dtype * (s * h + s * n_q_l * (nope_d + rope_d))}
    if op == "attn_mla_v_up_proj":
        flops = 2.0 * s * n_q_l * v_d * h
        return {"flops": flops, "bytes": dtype * (s * n_q_l * v_d + s * h)}
    if op == "attn_mla_prefill":
        seq = max(1, batch.num_prefill_tokens or s)
        flops = 4.0 * n_q_l * latent * seq * seq
        return {"flops": flops, "bytes": dtype * (seq * n_q_l * latent * 2)}
    if op == "attn_mla_decode":
        sd = max(0, batch.num_decode_tokens or (1 if batch.phase is BatchPhase.DECODE else 0))
        mean_ctx = max(
            1.0,
            batch.kv_cache_tokens / max(1, batch.num_decode_tokens or batch.batch_size),
        )
        flops = 4.0 * sd * n_q_l * latent * mean_ctx
        return {"flops": flops, "bytes": dtype * (sd * n_q_l * latent + sd * mean_ctx * latent)}
    return {"flops": 0.0, "bytes": 0.0}


def _dense_cost_for_op(
    *,
    op: str,
    variant: AttnVariant,
    model: ModelConfig,
    batch: BatchFeatures,
    tp: int,
) -> dict[str, Any]:
    if op == "attn_pre_proj":
        return _cost_pre_proj(variant=variant, model=model, batch=batch, tp=tp)
    if op == "attn_post_proj":
        return _cost_post_proj(variant=variant, model=model, batch=batch, tp=tp)
    if op == "attn_rope":
        return _cost_rope(variant=variant, model=model, batch=batch, tp=tp)
    if op == "attn_kv_cache_save":
        return _cost_kv_cache_save(variant=variant, model=model, batch=batch, tp=tp)
    if op in ("attn_prefill", "attn_decode"):
        return _cost_attn_core(
            variant=variant, model=model, batch=batch, tp=tp, which=op
        )
    return {"flops": 0.0, "bytes": 0.0}


@dataclass
class AttnOperator(Operator):
    """Single Frontier-aligned attention physical operator (1:1 TimeoutKernel)."""

    def __post_init__(self) -> None:
        self.kind = OperatorKind.COMPUTE_ATTN

    def expand_kernels(self):
        return self._single_kernel()


def make_attn_block_operators(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
) -> list[AttnOperator]:
    """Build the attention sub-DAG ops with Frontier names (no cross-deps wired)."""
    variant = model.resolved_attn_variant()
    ensure_attn_variant_supported(variant)
    tp = max(1, int(tp_size))
    op_names = attn_ops_for_variant(variant, batch.phase)
    ops: list[AttnOperator] = []
    for op_name in op_names:
        if variant is AttnVariant.MLA:
            cost = _mla_cost_for_op(op=op_name, model=model, batch=batch, tp=tp)
        else:
            cost = _dense_cost_for_op(
                op=op_name, variant=variant, model=model, batch=batch, tp=tp
            )
        ops.append(
            AttnOperator(
                name=f"L{layer_id}.{op_name}",
                kind=OperatorKind.COMPUTE_ATTN,
                variant=variant.value,
                phase=batch.phase,
                deps=[],  # wired by dag_builder
                features={
                    "rf_op": op_name,
                    "num_tokens": batch.num_tokens,
                    **cost,
                },
                layer_id=layer_id,
            )
        )
    # First op carries incoming deps; rest chained later.
    if ops:
        ops[0].deps = list(deps)
    return ops


# Backward-compatible alias used by older call sites / tests.
def make_attn_operator(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int = 1,
    expand_sub_kernels: bool = False,
) -> AttnOperator:
    """Return a single fused attention op (legacy); prefer ``make_attn_block_operators``."""
    del expand_sub_kernels
    ops = make_attn_block_operators(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=deps,
        tp_size=tp_size,
    )
    # Fuse costs into one operator for legacy callers.
    flops = sum(float(o.features.get("flops", 0.0)) for o in ops)
    nbytes = sum(float(o.features.get("bytes", 0.0)) for o in ops)
    name = (
        f"L{layer_id}.attn_prefill"
        if batch.phase is BatchPhase.PREFILL
        else f"L{layer_id}.attn_decode"
        if batch.phase is BatchPhase.DECODE
        else f"L{layer_id}.attn_mixed"
    )
    return AttnOperator(
        name=name,
        kind=OperatorKind.COMPUTE_ATTN,
        variant=model.resolved_attn_variant().value,
        phase=batch.phase,
        deps=list(deps),
        features={"flops": flops, "bytes": nbytes, "rf_op": "attn_fused"},
        layer_id=layer_id,
    )


def attention_compute_cost(
    *,
    variant: AttnVariant,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, Any]:
    """Aggregate flops/bytes across the RF-aligned attention block."""
    ensure_attn_variant_supported(variant)
    ops = make_attn_block_operators(
        layer_id=0,
        model=model,
        batch=batch,
        deps=[],
        tp_size=tp_size,
    )
    return {
        "flops": sum(float(o.features.get("flops", 0.0)) for o in ops),
        "bytes": sum(float(o.features.get("bytes", 0.0)) for o in ops),
    }
