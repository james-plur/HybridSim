"""Expected primitive-op short names for one transformer layer."""

from __future__ import annotations

from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchPhase,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm_names import (
    COMM_ATTN_TP_ALLREDUCE,
    COMM_EP_COMBINE,
    COMM_EP_DISPATCH,
    COMM_MLP_TP_ALLREDUCE,
    COMM_MOE_TP_ALLGATHER,
    COMM_MOE_TP_ALLREDUCE,
    COMM_SHARE_EXPERT_TP_ALLREDUCE,
)
from hybridsim_infer.workload_generators.types import AttnVariant


def expected_layer_primitives(
    *,
    attn_variant: AttnVariant,
    phase: BatchPhase,
    is_moe: bool = False,
    has_share_expert: bool = False,
    attn_tp: int = 1,
    moe_tp: int = 1,
    ep: int = 1,
    fused_add_norm: bool = False,
    num_prefill: int = 0,
    num_decode: int = 0,
) -> list[str]:
    if num_prefill <= 0 and num_decode <= 0:
        if phase is BatchPhase.DECODE:
            num_decode = 1
        elif phase is BatchPhase.MIXED:
            num_prefill = 1
            num_decode = 1
        else:
            num_prefill = 1

    names: list[str] = ["input_layernorm"]
    if attn_variant in (AttnVariant.MLA, AttnVariant.DSA):
        names.append("kv_cache_save")
        names.extend(["gemm_kv_up"] * num_prefill)
        names.extend(["gemm_q_lora", "gemm_q_expand"] * num_decode)
        names.extend(["fused_mla_attn"] * (num_prefill + num_decode))
        names.append("gemm_v_up")
        if attn_variant is AttnVariant.DSA:
            names.append("indexer_cache_save")
    else:
        names.extend(["gemm_qkv", "rope"])
        names.extend(["fused_attn"] * (num_prefill + num_decode))
        names.extend(["kv_cache_save", "gemm_o"])

    if attn_tp > 1:
        names.append(COMM_ATTN_TP_ALLREDUCE)
    if not fused_add_norm:
        names.append("add_attn_residual")
    names.append("post_attention_layernorm")

    if is_moe:
        names.extend(["moe_gating", "moe_topk"])
        if moe_tp > 1:
            names.append(COMM_MOE_TP_ALLGATHER)
        if ep > 1:
            names.append(COMM_EP_DISPATCH)
        names.extend(["moe_shuffle", "gemm_moe_up", "gemm_moe_down"])
        if ep > 1:
            names.append(COMM_EP_COMBINE)
        if has_share_expert:
            names.extend(["gemm_share_up", "share_act", "gemm_share_down"])
            if moe_tp > 1:
                names.append(COMM_SHARE_EXPERT_TP_ALLREDUCE)
        if moe_tp > 1:
            names.append(COMM_MOE_TP_ALLREDUCE)
    else:
        names.extend(["gemm_up", "mlp_act", "gemm_down"])
        if max(moe_tp, attn_tp) > 1:
            names.append(COMM_MLP_TP_ALLREDUCE)

    if not fused_add_norm:
        names.append("add_ffn_residual")
    return names
