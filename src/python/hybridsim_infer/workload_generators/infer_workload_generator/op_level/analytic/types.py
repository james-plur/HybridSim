"""Types for op-level Operator DAGs and kernel plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperatorKind(Enum):
    MEM = "mem"
    GEMM = "gemm"
    FUSED = "fused"
    COMM = "comm"


class CommCollective(Enum):
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALLGATHER = "allgather"
    DISPATCH = "dispatch"
    COMBINE = "combine"
    P2P = "p2p"


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


@dataclass
class KernelPlan:
    """One TimeoutKernel slot produced by lowering a mock Op."""

    name: str
    features: dict[str, Any] = field(default_factory=dict)
    kind: OperatorKind = OperatorKind.GEMM


@dataclass
class OperatorDAG:
    """Ordered list of mock Ops with cross-operator dependency indices."""

    operators: list[Any] = field(default_factory=list)

    def add(self, op: Any) -> int:
        idx = len(self.operators)
        self.operators.append(op)
        return idx

    def __len__(self) -> int:
        return len(self.operators)

    def op_names(self) -> list[str]:
        """Short names (strip ``L{id}.`` prefix)."""
        names: list[str] = []
        for op in self.operators:
            full = str(op.name)
            names.append(full.split(".", 1)[1] if "." in full else full)
        return names
