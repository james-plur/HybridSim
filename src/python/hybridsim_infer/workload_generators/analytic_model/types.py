"""Shared types for analytic Operator DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperatorKind(Enum):
    COMPUTE_ATTN = "compute_attn"
    COMPUTE_FFN = "compute_ffn"
    COMPUTE_MOE = "compute_moe"
    MEMORY = "memory"
    COMM = "comm"


class BatchPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class AttnVariant(Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"
    CSA = "csa"
    HSA = "hsa"
    DSA = "dsa"


class FfnActivation(Enum):
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    RELU = "relu"


class CommCollective(Enum):
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALLGATHER = "allgather"
    DISPATCH = "dispatch"
    COMBINE = "combine"
    P2P = "p2p"


class TpCommStyle(Enum):
    """How tensor-parallel communication is inserted after compute ops."""

    ALLREDUCE = "allreduce"
    RS_AG = "rs_ag"  # reduce_scatter then allgather


@dataclass
class BatchFeatures:
    """Token / phase summary extracted from a ScheduleBatch."""

    phase: BatchPhase
    num_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    batch_size: int
    #: Sum of KV context lengths across decode (and mixed) requests.
    kv_cache_tokens: int = 0


@dataclass
class KernelPlan:
    """One TimeoutKernel slot produced by expanding an Operator."""

    name: str
    #: Relative dependency indices within the same Operator expansion (local).
    local_deps: list[int] = field(default_factory=list)
    #: Work features used by OpAnalyzer (flops/bytes or payload/ranks).
    features: dict[str, Any] = field(default_factory=dict)
    kind: OperatorKind = OperatorKind.COMPUTE_ATTN


@dataclass
class OperatorDAG:
    """Ordered list of Operators with cross-operator dependency indices."""

    operators: list[Any] = field(default_factory=list)

    def add(self, op: Any) -> int:
        idx = len(self.operators)
        self.operators.append(op)
        return idx

    def __len__(self) -> int:
        return len(self.operators)

    def rf_op_names(self) -> list[str]:
        """Frontier short names (strip ``L{id}.`` prefix)."""
        from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
            strip_layer_prefix,
        )

        return [strip_layer_prefix(op.name) for op in self.operators]
