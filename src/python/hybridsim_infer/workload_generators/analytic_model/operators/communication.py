"""Communication operators aligned with Frontier ``comm`` family names."""

from __future__ import annotations

from dataclasses import dataclass

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
    COMM_ATTN_TP_ALLREDUCE,
    COMM_EP_COMBINE,
    COMM_EP_DISPATCH,
    COMM_MLP_TP_ALLREDUCE,
    COMM_MOE_TP_ALLGATHER,
    COMM_MOE_TP_ALLREDUCE,
    COMM_PP_SEND_RECV,
    COMM_SHARE_EXPERT_TP_ALLREDUCE,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    BatchPhase,
    CommCollective,
    OperatorKind,
)


def collective_volume_factor(collective: CommCollective, num_ranks: int) -> float:
    n = max(1, int(num_ranks))
    if n <= 1:
        return 0.0
    if collective is CommCollective.ALLREDUCE:
        return 2.0 * (n - 1) / n
    if collective in (
        CommCollective.REDUCE_SCATTER,
        CommCollective.ALLGATHER,
        CommCollective.DISPATCH,
        CommCollective.COMBINE,
    ):
        return (n - 1) / n
    if collective is CommCollective.P2P:
        return 1.0
    raise ValueError(f"Unknown collective: {collective!r}")


def hidden_state_payload_bytes(
    *,
    model: ModelConfig,
    batch: BatchFeatures,
) -> int:
    return int(batch.num_tokens) * int(model.hidden_size) * int(model.dtype_bytes)


@dataclass
class CommOperator(Operator):
    """Collective / P2P communication operator (1:1 TimeoutKernel)."""

    def __post_init__(self) -> None:
        self.kind = OperatorKind.COMM

    def expand_kernels(self):
        return self._single_kernel()


def make_comm_operator(
    *,
    rf_op: str,
    collective: CommCollective,
    layer_id: int,
    phase: BatchPhase,
    deps: list[int],
    payload_bytes: int,
    num_ranks: int,
) -> CommOperator:
    ranks = max(1, int(num_ranks))
    payload = max(0, int(payload_bytes))
    factor = collective_volume_factor(collective, ranks)
    return CommOperator(
        name=f"L{layer_id}.{rf_op}",
        kind=OperatorKind.COMM,
        variant=collective.value,
        phase=phase,
        deps=list(deps),
        features={
            "rf_op": rf_op,
            "payload_bytes": payload,
            "num_ranks": ranks,
            "volume_factor": factor,
            "collective": collective.value,
        },
        layer_id=layer_id,
    )


def make_attn_tp_allreduce(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
) -> CommOperator | None:
    if tp_size <= 1:
        return None
    return make_comm_operator(
        rf_op=COMM_ATTN_TP_ALLREDUCE,
        collective=CommCollective.ALLREDUCE,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=tp_size,
    )


def make_mlp_tp_allreduce(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
) -> CommOperator | None:
    if tp_size <= 1:
        return None
    return make_comm_operator(
        rf_op=COMM_MLP_TP_ALLREDUCE,
        collective=CommCollective.ALLREDUCE,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=tp_size,
    )


def make_moe_tp_allreduce(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
) -> CommOperator | None:
    if tp_size <= 1:
        return None
    return make_comm_operator(
        rf_op=COMM_MOE_TP_ALLREDUCE,
        collective=CommCollective.ALLREDUCE,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=tp_size,
    )


def make_moe_tp_allgather(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
) -> CommOperator | None:
    if tp_size <= 1:
        return None
    return make_comm_operator(
        rf_op=COMM_MOE_TP_ALLGATHER,
        collective=CommCollective.ALLGATHER,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=tp_size,
    )


def make_share_expert_tp_allreduce(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
) -> CommOperator | None:
    if tp_size <= 1:
        return None
    return make_comm_operator(
        rf_op=COMM_SHARE_EXPERT_TP_ALLREDUCE,
        collective=CommCollective.ALLREDUCE,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=tp_size,
    )


def make_ep_dispatch_combine(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps_before_dispatch: list[int],
    ep_size: int,
) -> tuple[CommOperator, CommOperator]:
    payload = hidden_state_payload_bytes(model=model, batch=batch)
    phase = batch.phase
    dispatch = make_comm_operator(
        rf_op=COMM_EP_DISPATCH,
        collective=CommCollective.DISPATCH,
        layer_id=layer_id,
        phase=phase,
        deps=list(deps_before_dispatch),
        payload_bytes=payload,
        num_ranks=ep_size,
    )
    combine = make_comm_operator(
        rf_op=COMM_EP_COMBINE,
        collective=CommCollective.COMBINE,
        layer_id=layer_id,
        phase=phase,
        deps=[],
        payload_bytes=payload,
        num_ranks=ep_size,
    )
    return dispatch, combine


def make_pp_p2p(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    direction: str = "send",
) -> CommOperator:
    """PP boundary uses Frontier name ``pipeline_parallel_send_recv``."""
    del direction
    return make_comm_operator(
        rf_op=COMM_PP_SEND_RECV,
        collective=CommCollective.P2P,
        layer_id=layer_id,
        phase=batch.phase,
        deps=list(deps),
        payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
        num_ranks=2,
    )


def make_tp_comm_ops(
    *,
    layer_id: int,
    model: ModelConfig,
    batch: BatchFeatures,
    deps: list[int],
    tp_size: int,
    style: str,
    tag: str,
) -> list[CommOperator]:
    """Legacy TP insert; maps to Frontier attn/mlp allreduce names."""
    if tp_size <= 1:
        return []
    rf = COMM_ATTN_TP_ALLREDUCE if tag == "attn" else COMM_MLP_TP_ALLREDUCE
    style_l = style.lower().strip()
    if style_l in ("allreduce", CommCollective.ALLREDUCE.value):
        op = make_comm_operator(
            rf_op=rf,
            collective=CommCollective.ALLREDUCE,
            layer_id=layer_id,
            phase=batch.phase,
            deps=list(deps),
            payload_bytes=hidden_state_payload_bytes(model=model, batch=batch),
            num_ranks=tp_size,
        )
        return [op]
    if style_l in ("rs_ag", "reduce_scatter_allgather"):
        payload = hidden_state_payload_bytes(model=model, batch=batch)
        rs = make_comm_operator(
            rf_op=f"{tag}_reduce_scatter",
            collective=CommCollective.REDUCE_SCATTER,
            layer_id=layer_id,
            phase=batch.phase,
            deps=list(deps),
            payload_bytes=payload,
            num_ranks=tp_size,
        )
        ag = make_comm_operator(
            rf_op=f"{tag}_allgather",
            collective=CommCollective.ALLGATHER,
            layer_id=layer_id,
            phase=batch.phase,
            deps=[],
            payload_bytes=payload,
            num_ranks=tp_size,
        )
        return [rs, ag]
    raise ValueError(f"Unsupported TP comm style: {style!r}")
