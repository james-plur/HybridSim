"""Build OperatorDAG from model / parallel config (Frontier RF-aligned)."""

from __future__ import annotations

from hybridsim_infer.workload_generators.analytic_model.configs import (
    ModelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.analytic_model.operators.attention import (
    ensure_attn_variant_supported,
    make_attn_block_operators,
)
from hybridsim_infer.workload_generators.analytic_model.operators.communication import (
    make_attn_tp_allreduce,
    make_ep_dispatch_combine,
    make_mlp_tp_allreduce,
    make_moe_tp_allgather,
    make_moe_tp_allreduce,
    make_pp_p2p,
    make_share_expert_tp_allreduce,
)
from hybridsim_infer.workload_generators.analytic_model.operators.ffn import (
    make_ffn_block_operators,
)
from hybridsim_infer.workload_generators.analytic_model.operators.memory import (
    make_memory_operator,
)
from hybridsim_infer.workload_generators.analytic_model.operators.moe import (
    make_moe_block_operators,
    make_share_expert_operators,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    OperatorDAG,
)


def _chain(dag: OperatorDAG, ops: list, prev: list[int]) -> list[int]:
    """Append operators in sequence; first uses ``prev``, rest chain to prior."""
    if not ops:
        return prev
    cur = prev
    last = prev[-1] if prev else -1
    for i, op in enumerate(ops):
        if i == 0:
            op.deps = list(cur)
        else:
            op.deps = [last]
        last = dag.add(op)
        cur = [last]
    return cur


def build_operator_dag(
    *,
    model: ModelConfig,
    parallel: ParallelConfig,
    batch: BatchFeatures,
    expand_attn_sub_kernels: bool = True,
    expand_ffn_sub_kernels: bool = True,
) -> OperatorDAG:
    """Construct a per-stage Transformer Operator DAG (RF-aligned op names).

    Dense layer::

        input_layernorm → attn_* → [attn_tp_allreduce] → [add_attn_residual]
        → post_attention_layernorm → mlp_* → [mlp_tp_allreduce] → [add_ffn_residual]

    MoE layer replaces ``mlp_*`` with gating / EP / grouped_gemm / share_expert.
    """
    del expand_attn_sub_kernels, expand_ffn_sub_kernels
    ensure_attn_variant_supported(model.resolved_attn_variant())

    dag = OperatorDAG()
    attn_tp = parallel.resolved_attn_tp()
    moe_tp = parallel.resolved_moe_tp()
    pp = max(1, int(parallel.pp_size))
    ep = max(1, int(parallel.ep_size))
    stage = max(0, min(int(parallel.pp_stage), pp - 1))
    n_layers = parallel.layers_on_stage(model.num_layers)
    fused = bool(model.fused_add_norm)

    prev: list[int] = []

    if pp > 1 and stage > 0:
        recv = make_pp_p2p(
            layer_id=stage * 1000,
            model=model,
            batch=batch,
            deps=[],
            direction="recv",
        )
        prev = [dag.add(recv)]

    for local_i in range(n_layers):
        layer_id = stage * max(1, (model.num_layers + pp - 1) // pp) + local_i
        prev = _append_layer(
            dag,
            layer_id=layer_id,
            model=model,
            batch=batch,
            prev=prev,
            attn_tp=attn_tp,
            moe_tp=moe_tp,
            ep=ep,
            fused_add_norm=fused,
        )

    if pp > 1 and stage < pp - 1:
        send = make_pp_p2p(
            layer_id=stage * 1000 + n_layers,
            model=model,
            batch=batch,
            deps=list(prev),
            direction="send",
        )
        dag.add(send)

    return dag


def _append_layer(
    dag: OperatorDAG,
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    prev: list[int],
    attn_tp: int,
    moe_tp: int,
    ep: int,
    fused_add_norm: bool,
) -> list[int]:
    ln = make_memory_operator(
        layer_id=layer_id,
        op_name="input_layernorm",
        model=model,
        batch=batch,
        deps=list(prev),
    )
    prev = [dag.add(ln)]

    attn_ops = make_attn_block_operators(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=attn_tp,
    )
    prev = _chain(dag, attn_ops, prev)

    ar = make_attn_tp_allreduce(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=attn_tp,
    )
    if ar is not None:
        prev = [dag.add(ar)]

    if not fused_add_norm:
        res = make_memory_operator(
            layer_id=layer_id,
            op_name="add_attn_residual",
            model=model,
            batch=batch,
            deps=list(prev),
        )
        prev = [dag.add(res)]

    pln = make_memory_operator(
        layer_id=layer_id,
        op_name="post_attention_layernorm",
        model=model,
        batch=batch,
        deps=list(prev),
    )
    prev = [dag.add(pln)]

    if model.is_moe:
        prev = _append_moe(
            dag,
            layer_id=layer_id,
            model=model,
            batch=batch,
            prev=prev,
            moe_tp=moe_tp,
            ep=ep,
        )
    else:
        ffn_ops = make_ffn_block_operators(
            layer_id=layer_id,
            model=model,
            batch=batch,
            deps=list(prev),
            tp_size=moe_tp if moe_tp > 1 else attn_tp,
        )
        prev = _chain(dag, ffn_ops, prev)
        mlp_ar = make_mlp_tp_allreduce(
            layer_id=layer_id,
            model=model,
            batch=batch,
            deps=list(prev),
            tp_size=max(moe_tp, attn_tp),
        )
        if mlp_ar is not None:
            prev = [dag.add(mlp_ar)]

    if not fused_add_norm:
        fres = make_memory_operator(
            layer_id=layer_id,
            op_name="add_ffn_residual",
            model=model,
            batch=batch,
            deps=list(prev),
        )
        prev = [dag.add(fres)]

    return prev


def _append_moe(
    dag: OperatorDAG,
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    prev: list[int],
    moe_tp: int,
    ep: int,
) -> list[int]:
    moe_ops = make_moe_block_operators(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=moe_tp,
    )
    gating = moe_ops[:2]
    rest = moe_ops[2:]
    prev = _chain(dag, gating, prev)

    ag = make_moe_tp_allgather(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=moe_tp,
    )
    if ag is not None:
        prev = [dag.add(ag)]

    if ep > 1:
        dispatch, combine = make_ep_dispatch_combine(
            layer_id=layer_id,
            model=model,
            batch=batch,
            deps_before_dispatch=prev,
            ep_size=ep,
        )
        d_idx = dag.add(dispatch)
        prev = _chain(dag, rest, [d_idx])
        combine.deps = list(prev)
        prev = [dag.add(combine)]
    else:
        prev = _chain(dag, rest, prev)

    share = make_share_expert_operators(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=moe_tp,
    )
    if share:
        prev = _chain(dag, share, prev)
        se_ar = make_share_expert_tp_allreduce(
            layer_id=layer_id,
            model=model,
            batch=batch,
            deps=list(prev),
            tp_size=moe_tp,
        )
        if se_ar is not None:
            prev = [dag.add(se_ar)]

    moe_ar = make_moe_tp_allreduce(
        layer_id=layer_id,
        model=model,
        batch=batch,
        deps=list(prev),
        tp_size=moe_tp,
    )
    if moe_ar is not None:
        prev = [dag.add(moe_ar)]
    return prev
