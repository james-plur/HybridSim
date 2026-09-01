"""Op DAG → kernel DAG analyzers.

An analyzer is one lowering strategy. Compute and communication can use
different analyzers: ``AnalyticAnalyzer`` turns GEMM / Mem / Fused (and, when
used alone, Comm) into TimeoutKernels; a comm analyzer such as
``RingCommAnalyzer`` turns only ``CommOp`` into Put/Wait primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class AnalyzeContext:
    """Per-DAG lowering context (rank is set when emitting a per-rank kernel DAG)."""

    workload_id: int = 0
    rank: int | None = None
    replica_id: int = 0
    num_ranks: int = 1


def op_kind(op: Any) -> str:
    kind = getattr(op, "kind", None)
    if kind is None:
        return ""
    value = getattr(kind, "value", kind)
    return str(value).lower().strip()


def is_comm_op(op: Any) -> bool:
    return op_kind(op) == "comm"


class OpAnalyzer(ABC):
    """One way to lower mock operators into Engine kernels."""

    def handles(self, op: Any) -> bool:
        """Whether ``lower_op`` should be used for this operator."""
        return True

    @abstractmethod
    def lower_op(
        self,
        op: Any,
        *,
        op_index: int,
        ctx: AnalyzeContext,
    ) -> list[dict[str, Any]]:
        """Return kernels for ``op``.

        Use ``rel_deps`` for intra-op kernel indices. Empty ``rel_deps`` (or
        omitted) attaches the kernels to all kernels of predecessor operators.
        An empty list skips the operator.
        """

    def analyze(
        self,
        op_dag: Any,
        *,
        workload_id: int,
        rank: int | None = None,
        replica_id: int = 0,
        num_ranks: int = 1,
    ) -> dict[str, Any]:
        ctx = AnalyzeContext(
            workload_id=int(workload_id),
            rank=rank,
            replica_id=int(replica_id),
            num_ranks=max(1, int(num_ranks)),
        )
        return analyze_operator_dag(
            op_dag,
            workload_id=int(workload_id),
            lower=lambda op, i: self.lower_op(op, op_index=i, ctx=ctx),
        )


def analyze_operator_dag(
    op_dag: Any,
    *,
    workload_id: int,
    lower: Callable[[Any, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Walk ``op_dag`` in order, remap op deps onto produced kernel indices."""
    kernels: list[dict[str, Any]] = []
    op_kernel_ids: list[list[int]] = []
    for op_idx, op in enumerate(op_dag.operators):
        incoming: list[int] = []
        seen_in: set[int] = set()
        for dep_op in getattr(op, "deps", []):
            if dep_op < 0 or dep_op >= op_idx:
                raise ValueError(
                    f"Operator {getattr(op, 'name', op_idx)!r} has invalid dep "
                    f"{dep_op} (must be earlier operator index)"
                )
            for kid in op_kernel_ids[dep_op]:
                if kid not in seen_in:
                    seen_in.add(kid)
                    incoming.append(kid)
        prims = lower(op, op_idx)
        if not prims:
            op_kernel_ids.append([])
            continue
        base = len(kernels)
        new_ids: list[int] = []
        for prim in prims:
            rel = prim.get("rel_deps")
            if rel:
                deps = [base + int(d) for d in rel]
            else:
                deps = list(incoming)
            entry: dict[str, Any] = {
                "name": prim["name"],
                "duration": float(prim.get("duration", 0.0)),
                "dependencies": deps,
            }
            if "type" in prim:
                ktype = int(prim["type"])
                if ktype != 0:
                    entry["type"] = ktype
            if prim.get("params"):
                entry["params"] = dict(prim["params"])
            new_ids.append(len(kernels))
            kernels.append(entry)
        op_kernel_ids.append(new_ids)
    return {
        "workload_id": int(workload_id),
        "kernels": kernels,
    }


def analyze_split(
    op_dag: Any,
    *,
    compute: OpAnalyzer,
    comm: Optional[OpAnalyzer] = None,
    workload_id: int,
    rank: int | None = None,
    replica_id: int = 0,
    num_ranks: int = 1,
) -> dict[str, Any]:
    """Lower compute ops with ``compute`` and CommOps with ``comm`` when set."""
    ctx = AnalyzeContext(
        workload_id=int(workload_id),
        rank=rank,
        replica_id=int(replica_id),
        num_ranks=max(1, int(num_ranks)),
    )

    def lower(op: Any, op_index: int) -> list[dict[str, Any]]:
        if comm is not None and comm.handles(op):
            return comm.lower_op(op, op_index=op_index, ctx=ctx)
        return compute.lower_op(op, op_index=op_index, ctx=ctx)

    return analyze_operator_dag(
        op_dag,
        workload_id=int(workload_id),
        lower=lower,
    )
