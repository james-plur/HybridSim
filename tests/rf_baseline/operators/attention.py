"""Frozen attention rf_op costs (dense / MLA families)."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.types import (
    AttnVariant,
    ensure_attn_variant_supported,
)


def _indexer_cache_save_cost(
    model: ModelConfig, batch: BatchFeatures
) -> dict[str, Any]:
    s = max(0, int(batch.num_tokens))
    idx_hd = max(0, int(getattr(model, "index_head_dim", 0) or 0))
    idx_b = int(getattr(model, "index_dtype_bytes", 0) or 0) or max(
        1, int(model.dtype_bytes)
    )
    return {"flops": 0.0, "bytes": float(idx_b * s * idx_hd)}


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
    flops = 2.0 * s * h * (n_q_l * d)
    flops += 2.0 * s * h * (n_kv_l * d) * 2.0
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
        flops = 0.0
        nbytes = 0.0
        for tokens_i, ctx_i in batch.iter_decode_attn_pairs():
            flops += 4.0 * tokens_i * n_q_l * d * ctx_i
            nbytes += dtype * (
                tokens_i * n_q_l * d
                + tokens_i * ctx_i * n_kv_l * d * 2
                + tokens_i * n_q_l * d
            )
        return {"flops": flops, "bytes": nbytes}

    flops = 0.0
    nbytes = 0.0
    for chunk_i, cached_i in batch.iter_prefill_attn_pairs():
        ctx_i = max(chunk_i, cached_i + chunk_i)
        flops += 4.0 * n_q_l * d * chunk_i * ctx_i
        nbytes += dtype * (
            chunk_i * n_q_l * d + ctx_i * n_kv_l * d * 2 + chunk_i * n_q_l * d
        )
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
        flops = 0.0
        nbytes = 0.0
        for chunk_i, cached_i in batch.iter_prefill_attn_pairs():
            ctx_i = max(chunk_i, cached_i + chunk_i)
            flops += 4.0 * n_q_l * latent * chunk_i * ctx_i
            nbytes += dtype * (
                chunk_i * n_q_l * latent + ctx_i * latent + chunk_i * n_q_l * latent
            )
        return {"flops": flops, "bytes": nbytes}
    if op == "attn_mla_decode":
        flops = 0.0
        nbytes = 0.0
        for tokens_i, ctx_i in batch.iter_decode_attn_pairs():
            flops += 4.0 * tokens_i * n_q_l * latent * ctx_i
            nbytes += dtype * (tokens_i * n_q_l * latent + tokens_i * ctx_i * latent)
        return {"flops": flops, "bytes": nbytes}
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


def attn_cost_for_op(
    *,
    op_name: str,
    model: ModelConfig,
    batch: BatchFeatures,
    tp_size: int = 1,
) -> dict[str, Any]:
    variant = model.resolved_attn_variant()
    ensure_attn_variant_supported(variant)
    tp = max(1, int(tp_size))
    if op_name == "attn_indexer_cache_save":
        return _indexer_cache_save_cost(model, batch)
    if variant in (AttnVariant.MLA, AttnVariant.DSA):
        return _mla_cost_for_op(op=op_name, model=model, batch=batch, tp=tp)
    return _dense_cost_for_op(
        op=op_name, variant=variant, model=model, batch=batch, tp=tp
    )
