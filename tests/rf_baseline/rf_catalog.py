"""Frozen Frontier RF operator-name lists for baseline aggregation in tests."""

from __future__ import annotations

from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchPhase,
)
from hybridsim_infer.workload_generators.types import AttnVariant

DENSE_ATTN_PREFILL_OPS: tuple[str, ...] = (
    "attn_pre_proj",
    "attn_rope",
    "attn_prefill",
    "attn_kv_cache_save",
    "attn_post_proj",
)

DENSE_ATTN_DECODE_OPS: tuple[str, ...] = (
    "attn_pre_proj",
    "attn_rope",
    "attn_decode",
    "attn_kv_cache_save",
    "attn_post_proj",
)

DENSE_ATTN_MIXED_OPS: tuple[str, ...] = (
    "attn_pre_proj",
    "attn_rope",
    "attn_prefill",
    "attn_decode",
    "attn_kv_cache_save",
    "attn_post_proj",
)

MLA_ATTN_PREFILL_OPS: tuple[str, ...] = (
    "attn_mla_kv_cache_save",
    "attn_mla_prefill_kv_up_proj",
    "attn_mla_prefill",
    "attn_mla_v_up_proj",
)

MLA_ATTN_DECODE_OPS: tuple[str, ...] = (
    "attn_mla_kv_cache_save",
    "attn_mla_decode_q_latent_proj",
    "attn_mla_decode",
    "attn_mla_v_up_proj",
)

MLA_ATTN_MIXED_OPS: tuple[str, ...] = (
    "attn_mla_kv_cache_save",
    "attn_mla_prefill_kv_up_proj",
    "attn_mla_prefill",
    "attn_mla_decode_q_latent_proj",
    "attn_mla_decode",
    "attn_mla_v_up_proj",
)

FFN_OPS: tuple[str, ...] = ("mlp_up_proj", "mlp_act", "mlp_down_proj")

SHARE_EXPERT_OPS: tuple[str, ...] = (
    "share_expert_up_proj",
    "share_expert_act",
    "share_expert_down_proj",
)


def dense_attn_ops_for_phase(phase: BatchPhase) -> tuple[str, ...]:
    if phase is BatchPhase.PREFILL:
        return DENSE_ATTN_PREFILL_OPS
    if phase is BatchPhase.DECODE:
        return DENSE_ATTN_DECODE_OPS
    return DENSE_ATTN_MIXED_OPS


def mla_attn_ops_for_phase(phase: BatchPhase) -> tuple[str, ...]:
    if phase is BatchPhase.PREFILL:
        return MLA_ATTN_PREFILL_OPS
    if phase is BatchPhase.DECODE:
        return MLA_ATTN_DECODE_OPS
    return MLA_ATTN_MIXED_OPS


def attn_ops_for_variant(
    variant: AttnVariant, phase: BatchPhase
) -> tuple[str, ...]:
    if variant is AttnVariant.MLA:
        return mla_attn_ops_for_phase(phase)
    if variant is AttnVariant.DSA:
        return mla_attn_ops_for_phase(phase) + ("attn_indexer_cache_save",)
    return dense_attn_ops_for_phase(phase)
