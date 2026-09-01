"""Comm analyzer: lower CommOp into per-rank Put / Signal / Wait / Get kernels."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analyzer import (
    AnalyzeContext,
    OpAnalyzer,
    is_comm_op,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    CommCollective,
)

KERNEL_TIMEOUT = 0
KERNEL_PUT = 1
KERNEL_SIGNAL = 2
KERNEL_WAIT = 3
KERNEL_GET = 4


def encode_conn(op_index: int, step: int, sender_rank: int, extra: int = 0) -> int:
    """Stable connection id shared by a Put and the matching Wait."""
    return (
        (int(op_index) << 42)
        | (int(step) << 28)
        | (int(sender_rank) << 14)
        | int(extra)
    )


def addr_str(replica_id: int, rank: int) -> str:
    return f"{int(replica_id)}:{int(rank)}"


def ranks_per_replica(parallel: Any, override: int = 0) -> int:
    if int(override) > 0:
        return int(override)
    attn = int(parallel.resolved_attn_tp()) if parallel is not None else 1
    moe = int(parallel.resolved_moe_tp()) if parallel is not None else 1
    ep = int(getattr(parallel, "ep_size", 1) or 1) if parallel is not None else 1
    return max(1, attn, moe, ep)


def _put(
    name: str,
    dst: str,
    conn_id: int,
    payload_bytes: float,
    rel_deps: list[int],
    qos: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": KERNEL_PUT,
        "duration": 0.0,
        "rel_deps": list(rel_deps),
        "params": {
            "dst_addr": dst,
            "conn_id": int(conn_id),
            "qos": int(qos),
            "payload_bytes": float(payload_bytes),
        },
    }


def _wait(
    name: str,
    conn_id: int,
    rel_deps: list[int],
    payload_bytes: float = 0.0,
    qos: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": KERNEL_WAIT,
        "duration": 0.0,
        "rel_deps": list(rel_deps),
        "params": {
            "conn_id": int(conn_id),
            "qos": int(qos),
            "payload_bytes": float(payload_bytes),
        },
    }


class RingCommAnalyzer(OpAnalyzer):
    """Lower a CommOp into ring / P2P / all-to-all primitives for one rank.

    Does not handle compute ops; pair with ``AnalyticAnalyzer`` via
    ``analyze_split``.
    """

    def __init__(
        self,
        *,
        replica_id: int = 0,
        num_ranks: int = 1,
        qos: int = 0,
    ) -> None:
        self.replica_id = int(replica_id)
        self.num_ranks = max(1, int(num_ranks))
        self.qos = int(qos)

    def handles(self, op: Any) -> bool:
        return is_comm_op(op)

    def lower_op(
        self,
        op: Any,
        *,
        op_index: int,
        ctx: AnalyzeContext,
    ) -> list[dict[str, Any]]:
        rank = ctx.rank if ctx.rank is not None else 0
        replica_id = ctx.replica_id if ctx.replica_id else self.replica_id
        return self.expand(
            op,
            rank=int(rank),
            op_index=int(op_index),
            replica_id=int(replica_id),
        )

    def expand(
        self,
        op: Any,
        *,
        rank: int,
        op_index: int,
        replica_id: int | None = None,
    ) -> list[dict[str, Any]]:
        feats = op.features() if hasattr(op, "features") else {}
        n = max(1, int(feats.get("num_ranks", self.num_ranks) or self.num_ranks))
        payload = float(feats.get("payload_bytes", 0.0) or 0.0)
        collective = feats.get("collective", CommCollective.ALLREDUCE.value)
        if isinstance(collective, CommCollective):
            collective = collective.value
        name = str(getattr(op, "name", f"comm{op_index}"))
        r = int(rank)
        rid = self.replica_id if replica_id is None else int(replica_id)
        if n <= 1:
            return []
        if collective in (
            CommCollective.ALLREDUCE.value,
            "allreduce",
        ):
            return self._ring(name, op_index, r, n, payload, rid, rounds=2)
        if collective in (
            CommCollective.REDUCE_SCATTER.value,
            "reduce_scatter",
        ):
            return self._ring(name, op_index, r, n, payload, rid, rounds=1)
        if collective in (CommCollective.ALLGATHER.value, "allgather"):
            return self._ring(name, op_index, r, n, payload, rid, rounds=1)
        if collective in (
            CommCollective.DISPATCH.value,
            CommCollective.COMBINE.value,
            "dispatch",
            "combine",
        ):
            return self._all_to_all(name, op_index, r, n, payload, rid)
        return self._p2p(name, op_index, r, n, payload, rid)

    def _ring(
        self,
        name: str,
        op_index: int,
        rank: int,
        n: int,
        payload: float,
        replica_id: int,
        *,
        rounds: int,
    ) -> list[dict[str, Any]]:
        chunk = payload / float(n) if n else 0.0
        steps = (n - 1) * int(rounds)
        kernels: list[dict[str, Any]] = []
        prev: list[int] = []
        send_to = (rank + 1) % n
        recv_from = (rank - 1 + n) % n
        dst = addr_str(replica_id, send_to)
        for s in range(steps):
            put_i = len(kernels)
            wait_i = put_i + 1
            kernels.append(
                _put(
                    f"{name}.s{s}.put",
                    dst,
                    encode_conn(op_index, s, rank),
                    chunk,
                    prev,
                    qos=self.qos,
                )
            )
            kernels.append(
                _wait(
                    f"{name}.s{s}.wait",
                    encode_conn(op_index, s, recv_from),
                    prev,
                    payload_bytes=chunk,
                    qos=self.qos,
                )
            )
            prev = [put_i, wait_i]
        return kernels

    def _all_to_all(
        self,
        name: str,
        op_index: int,
        rank: int,
        n: int,
        payload: float,
        replica_id: int,
    ) -> list[dict[str, Any]]:
        chunk = payload / float(n) if n else 0.0
        kernels: list[dict[str, Any]] = []
        for peer in range(n):
            if peer == rank:
                continue
            kernels.append(
                _put(
                    f"{name}.to{peer}.put",
                    addr_str(replica_id, peer),
                    encode_conn(op_index, 0, rank, peer),
                    chunk,
                    [],
                    qos=self.qos,
                )
            )
            kernels.append(
                _wait(
                    f"{name}.from{peer}.wait",
                    encode_conn(op_index, 0, peer, rank),
                    [],
                    payload_bytes=chunk,
                    qos=self.qos,
                )
            )
        return kernels

    def _p2p(
        self,
        name: str,
        op_index: int,
        rank: int,
        n: int,
        payload: float,
        replica_id: int,
    ) -> list[dict[str, Any]]:
        sender = 0
        receiver = 1 if n > 1 else 0
        conn = encode_conn(op_index, 0, sender, receiver)
        if rank == sender:
            return [
                _put(
                    f"{name}.put",
                    addr_str(replica_id, receiver),
                    conn,
                    payload,
                    [],
                    qos=self.qos,
                )
            ]
        if rank == receiver:
            return [
                _wait(
                    f"{name}.wait",
                    conn,
                    [],
                    payload_bytes=payload,
                    qos=self.qos,
                )
            ]
        return []


# Backward-compatible name used in older tests / call sites.
RingCommParser = RingCommAnalyzer
