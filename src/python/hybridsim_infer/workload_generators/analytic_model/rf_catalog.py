"""Frontier RF-aligned operator name catalog (structural parity).

Names match Frontier ``OperatorSpec.name`` / ``execution_time_attr`` sources so
analytical DAGs can be checked against RF block composition.
"""

from __future__ import annotations

from hybridsim_infer.workload_generators.analytic_model.types import (
    AttnVariant,
    BatchPhase,
)

# Dense attention physical ops (Frontier dense_attention family + linear ops).
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

# Mixed: both kernels (matches RF predicting both components on mixed batches).
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

MEMORY_PREFIX_OPS: tuple[str, ...] = ("input_layernorm",)
MEMORY_MID_OPS: tuple[str, ...] = ("add_attn_residual", "post_attention_layernorm")
MEMORY_SUFFIX_OPS: tuple[str, ...] = ("add_ffn_residual",)

FFN_OPS: tuple[str, ...] = ("mlp_up_proj", "mlp_act", "mlp_down_proj")

MOE_OPS: tuple[str, ...] = (
    "moe_gating_linear",
    "moe_gating_routing_topk",
    "moe_shuffling",
    "moe_grouped_gemm",
)

SHARE_EXPERT_OPS: tuple[str, ...] = (
    "share_expert_up_proj",
    "share_expert_act",
    "share_expert_down_proj",
)

# Communication (Frontier CommOperatorSpec names).
COMM_ATTN_TP_ALLREDUCE = "attn_tensor_parallel_allreduce"
COMM_MLP_TP_ALLREDUCE = "mlp_tensor_parallel_allreduce"
COMM_MOE_TP_ALLREDUCE = "moe_tensor_parallel_allreduce"
COMM_MOE_TP_ALLGATHER = "moe_tensor_parallel_allgather"
COMM_SHARE_EXPERT_TP_ALLREDUCE = "share_expert_tensor_parallel_allreduce"
COMM_EP_DISPATCH = "expert_parallel_alltoall_dispatch"
COMM_EP_COMBINE = "expert_parallel_alltoall_combine"
COMM_PP_SEND_RECV = "pipeline_parallel_send_recv"


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
    # MHA / GQA / MQA share dense family.
    return dense_attn_ops_for_phase(phase)


def expected_layer_op_names(
    *,
    attn_variant: AttnVariant,
    phase: BatchPhase,
    is_moe: bool = False,
    has_share_expert: bool = False,
    attn_tp: int = 1,
    moe_tp: int = 1,
    ep: int = 1,
    fused_add_norm: bool = False,
) -> list[str]:
    """Expected Frontier-aligned op short-names for one transformer layer."""
    names: list[str] = []
    names.extend(MEMORY_PREFIX_OPS)
    names.extend(attn_ops_for_variant(attn_variant, phase))
    if attn_tp > 1:
        names.append(COMM_ATTN_TP_ALLREDUCE)
    if not fused_add_norm:
        names.append("add_attn_residual")
    names.append("post_attention_layernorm")

    if is_moe:
        names.extend(("moe_gating_linear", "moe_gating_routing_topk"))
        if moe_tp > 1:
            names.append(COMM_MOE_TP_ALLGATHER)
        if ep > 1:
            names.append(COMM_EP_DISPATCH)
        names.extend(("moe_shuffling", "moe_grouped_gemm"))
        if ep > 1:
            names.append(COMM_EP_COMBINE)
        if has_share_expert:
            names.extend(SHARE_EXPERT_OPS)
            if moe_tp > 1:
                names.append(COMM_SHARE_EXPERT_TP_ALLREDUCE)
        if moe_tp > 1:
            names.append(COMM_MOE_TP_ALLREDUCE)
    else:
        names.extend(FFN_OPS)
        if moe_tp > 1 or attn_tp > 1:
            # Dense FFN uses mlp TP; fall back to attn_tp when moe_tp unset.
            tp = max(moe_tp, attn_tp)
            if tp > 1:
                names.append(COMM_MLP_TP_ALLREDUCE)

    if not fused_add_norm:
        names.append("add_ffn_residual")
    return names


def strip_layer_prefix(full_name: str) -> str:
    """``L12.attn_prefill`` → ``attn_prefill``; PP ops keep full short name."""
    if "." in full_name:
        return full_name.split(".", 1)[1]
    return full_name
