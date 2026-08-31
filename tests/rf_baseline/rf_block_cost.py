"""Sum frozen rf_op flops/bytes by block for comparison with the primitive DAG."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.lower import (
    lower_op,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    CommCollective,
    OperatorDAG,
    OperatorKind,
    collective_volume_factor,
)
from rf_baseline.operators.attention import attn_cost_for_op
from rf_baseline.operators.communication import hidden_state_payload_bytes
from rf_baseline.operators.ffn import ffn_cost_for_op
from rf_baseline.operators.memory import memory_cost_for_op
from rf_baseline.operators.moe import moe_cost_for_op
from rf_baseline.rf_catalog import FFN_OPS, SHARE_EXPERT_OPS, attn_ops_for_variant

_ATTN_PROJ = frozenset(
    {
        "attn_pre_proj",
        "attn_post_proj",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_v_up_proj",
        "gemm_qkv",
        "gemm_o",
        "gemm_kv_up",
        "gemm_q_lora",
        "gemm_q_expand",
        "gemm_v_up",
    }
)
_ATTN_CORE = frozenset(
    {
        "attn_prefill",
        "attn_decode",
        "attn_mla_prefill",
        "attn_mla_decode",
        "fused_attn",
        "fused_mla_attn",
    }
)
_ATTN_SIDE = frozenset(
    {
        "attn_rope",
        "attn_kv_cache_save",
        "attn_mla_kv_cache_save",
        "attn_indexer_cache_save",
        "rope",
        "kv_cache_save",
        "indexer_cache_save",
    }
)
_LAYER_MEM = frozenset(
    {
        "input_layernorm",
        "add_attn_residual",
        "post_attention_layernorm",
        "add_ffn_residual",
    }
)
_FFN = frozenset({"gemm_up", "mlp_act", "gemm_down"})
_MOE = frozenset(
    {
        "moe_gating",
        "moe_topk",
        "moe_shuffle",
        "gemm_moe_up",
        "gemm_moe_down",
        "gemm_share_up",
        "share_act",
        "gemm_share_down",
    }
)


def _short(name: str) -> str:
    return name.split(".", 1)[1] if "." in name else name


def _empty() -> dict[str, float]:
    return {"flops": 0.0, "bytes": 0.0}


def _add(dst: dict[str, float], src: dict[str, Any]) -> None:
    dst["flops"] += float(src.get("flops", 0.0))
    dst["bytes"] += float(src.get("bytes", 0.0))


def _block_for_name(name: str) -> str:
    short = _short(name)
    if short in _ATTN_PROJ:
        return "attn_proj"
    if short in _ATTN_CORE:
        return "attn_core"
    if short in _ATTN_SIDE:
        return "attn_side"
    if short in _LAYER_MEM:
        return "layer_mem"
    if short in _FFN:
        return "ffn"
    if short in _MOE:
        return "moe"
    return "comm"


def _add_comm(
    blocks: dict[str, dict[str, float]],
    *,
    model: ModelConfig,
    batch: BatchFeatures,
    collective: CommCollective,
    num_ranks: int,
) -> None:
    ranks = max(1, int(num_ranks))
    if ranks <= 1:
        return
    payload = hidden_state_payload_bytes(model=model, batch=batch)
    factor = collective_volume_factor(collective, ranks)
    blocks["comm"]["bytes"] += float(payload) * float(factor)


def rf_block_cost(
    *,
    model: ModelConfig,
    parallel: ParallelConfig,
    batch: BatchFeatures,
) -> dict[str, dict[str, float]]:
    """Aggregate frozen rf_op costs over the same PP-stage layers as the mock DAG."""
    blocks = {
        "attn_proj": _empty(),
        "attn_core": _empty(),
        "attn_side": _empty(),
        "ffn": _empty(),
        "moe": _empty(),
        "layer_mem": _empty(),
        "comm": _empty(),
    }
    attn_tp = parallel.resolved_attn_tp()
    moe_tp = parallel.resolved_moe_tp()
    ep = max(1, int(parallel.ep_size))
    pp = max(1, int(parallel.pp_size))
    stage = max(0, min(int(parallel.pp_stage), pp - 1))
    n_layers = parallel.layers_on_stage(model.num_layers)
    stride = max(1, (int(model.num_layers) + pp - 1) // pp)
    variant = model.resolved_attn_variant()
    fused = bool(model.fused_add_norm)

    if pp > 1 and stage > 0:
        _add_comm(
            blocks,
            model=model,
            batch=batch,
            collective=CommCollective.P2P,
            num_ranks=2,
        )

    for local_i in range(n_layers):
        layer_id = stage * stride + local_i
        use_moe = model.layer_is_moe(layer_id)
        for name in attn_ops_for_variant(variant, batch.phase):
            cost = attn_cost_for_op(
                op_name=name, model=model, batch=batch, tp_size=attn_tp
            )
            if name in _ATTN_PROJ:
                _add(blocks["attn_proj"], cost)
            elif name in _ATTN_CORE:
                _add(blocks["attn_core"], cost)
            elif name in _ATTN_SIDE:
                if name == "attn_rope":
                    _add(
                        blocks["attn_side"],
                        {"flops": 0.0, "bytes": cost.get("bytes", 0.0)},
                    )
                else:
                    _add(blocks["attn_side"], cost)
        if attn_tp > 1:
            _add_comm(
                blocks,
                model=model,
                batch=batch,
                collective=CommCollective.ALLREDUCE,
                num_ranks=attn_tp,
            )
        mem_names = ["input_layernorm", "post_attention_layernorm"]
        if not fused:
            mem_names.extend(["add_attn_residual", "add_ffn_residual"])
        for name in mem_names:
            _add(blocks["layer_mem"], memory_cost_for_op(op_name=name, model=model, batch=batch))

        if use_moe:
            for name in (
                "moe_gating_linear",
                "moe_gating_routing_topk",
                "moe_shuffling",
                "moe_grouped_gemm",
            ):
                cost = moe_cost_for_op(
                    op_name=name, model=model, batch=batch, tp_size=moe_tp
                )
                if name in ("moe_gating_routing_topk", "moe_shuffling"):
                    _add(blocks["moe"], {"flops": 0.0, "bytes": cost.get("bytes", 0.0)})
                else:
                    _add(blocks["moe"], cost)
            if model.has_share_expert():
                for name in SHARE_EXPERT_OPS:
                    cost = moe_cost_for_op(
                        op_name=name, model=model, batch=batch, tp_size=moe_tp
                    )
                    if name.endswith("_act"):
                        _add(
                            blocks["moe"],
                            {"flops": 0.0, "bytes": cost.get("bytes", 0.0)},
                        )
                    else:
                        _add(blocks["moe"], cost)
            if moe_tp > 1:
                _add_comm(
                    blocks,
                    model=model,
                    batch=batch,
                    collective=CommCollective.ALLGATHER,
                    num_ranks=moe_tp,
                )
                _add_comm(
                    blocks,
                    model=model,
                    batch=batch,
                    collective=CommCollective.ALLREDUCE,
                    num_ranks=moe_tp,
                )
                if model.has_share_expert():
                    _add_comm(
                        blocks,
                        model=model,
                        batch=batch,
                        collective=CommCollective.ALLREDUCE,
                        num_ranks=moe_tp,
                    )
            if ep > 1:
                _add_comm(
                    blocks,
                    model=model,
                    batch=batch,
                    collective=CommCollective.DISPATCH,
                    num_ranks=ep,
                )
                _add_comm(
                    blocks,
                    model=model,
                    batch=batch,
                    collective=CommCollective.COMBINE,
                    num_ranks=ep,
                )
        else:
            ffn_tp = moe_tp if moe_tp > 1 else attn_tp
            for name in FFN_OPS:
                cost = ffn_cost_for_op(
                    op_name=name, model=model, batch=batch, tp_size=ffn_tp
                )
                if name == "mlp_act":
                    _add(blocks["ffn"], {"flops": 0.0, "bytes": cost.get("bytes", 0.0)})
                else:
                    _add(blocks["ffn"], cost)
            if max(moe_tp, attn_tp) > 1:
                _add_comm(
                    blocks,
                    model=model,
                    batch=batch,
                    collective=CommCollective.ALLREDUCE,
                    num_ranks=max(moe_tp, attn_tp),
                )

    if pp > 1 and stage < pp - 1:
        _add_comm(
            blocks,
            model=model,
            batch=batch,
            collective=CommCollective.P2P,
            num_ranks=2,
        )
    return blocks


def sum_dag_blocks(dag: OperatorDAG) -> dict[str, dict[str, float]]:
    """Sum primitive-DAG flops/bytes (comm: payload * volume_factor)."""
    blocks = {
        "attn_proj": _empty(),
        "attn_core": _empty(),
        "attn_side": _empty(),
        "ffn": _empty(),
        "moe": _empty(),
        "layer_mem": _empty(),
        "comm": _empty(),
    }
    for op in dag.operators:
        plan = lower_op(op)
        key = _block_for_name(op.name)
        if plan.kind is OperatorKind.COMM:
            moved = float(plan.features.get("payload_bytes", 0.0)) * float(
                plan.features.get("volume_factor", 0.0)
            )
            blocks["comm"]["bytes"] += moved
            continue
        _add(blocks[key], plan.features)
    return blocks
