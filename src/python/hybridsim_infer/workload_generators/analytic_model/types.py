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
    """Token / phase summary extracted from a ScheduleBatch.

    Attention costing is **per-request** (FlashAttention-varlen style): use the
    ``prefill_*_lens`` / ``decode_*_lens`` lists. Scalar fields are aggregates for
    linear/FFN/comm (packed token counts) and convenience.

    ``cached_decode_tokens`` counts **decode-only** context sums.
    ``cached_prefix_tokens`` counts **prefill-only** already-cached prefix sums.
    """

    phase: BatchPhase
    num_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    batch_size: int
    #: Sum of decode KV context lengths (decode requests only).
    cached_decode_tokens: int = 0
    #: Sum of prefill already-cached prefix lengths (prefill requests only).
    cached_prefix_tokens: int = 0
    #: Per prefill request: tokens scheduled this step (no pad).
    prefill_chunk_lens: list[int] = field(default_factory=list)
    #: Per prefill request: cached prefix before this step.
    prefill_cached_lens: list[int] = field(default_factory=list)
    #: Per decode request: tokens scheduled this step.
    decode_token_lens: list[int] = field(default_factory=list)
    #: Per decode request: KV context length visible to attention.
    decode_kv_lens: list[int] = field(default_factory=list)

    def iter_prefill_attn_pairs(self) -> list[tuple[int, int]]:
        """Return ``(chunk_i, cached_i)`` for each prefill request.

        Falls back to a single pair from scalar aggregates when lists are empty
        (hand-built ``BatchFeatures`` in tests).
        """
        if self.prefill_chunk_lens:
            cached = self.prefill_cached_lens
            if len(cached) < len(self.prefill_chunk_lens):
                cached = list(cached) + [0] * (
                    len(self.prefill_chunk_lens) - len(cached)
                )
            return [
                (max(0, int(c)), max(0, int(k)))
                for c, k in zip(self.prefill_chunk_lens, cached)
                if int(c) > 0
            ]
        chunk = max(0, int(self.num_prefill_tokens) or 0)
        if chunk <= 0:
            return []
        cached = max(0, int(self.cached_prefix_tokens or 0))
        return [(chunk, cached)]

    def iter_decode_attn_pairs(self) -> list[tuple[int, int]]:
        """Return ``(tokens_i, kv_ctx_i)`` for each decode request."""
        if self.decode_token_lens:
            kvs = self.decode_kv_lens
            if len(kvs) < len(self.decode_token_lens):
                kvs = list(kvs) + [0] * (len(self.decode_token_lens) - len(kvs))
            return [
                (max(0, int(t)), max(1, int(k)))
                for t, k in zip(self.decode_token_lens, kvs)
                if int(t) > 0
            ]
        tokens = max(0, int(self.num_decode_tokens) or 0)
        if tokens <= 0 and self.phase is BatchPhase.DECODE:
            tokens = 1
        if tokens <= 0:
            return []
        # Scalar fallback: distribute total KV evenly across decode tokens.
        n_req = max(1, int(self.batch_size) if self.phase is BatchPhase.DECODE else 1)
        if self.decode_kv_lens:
            # Should not happen if token lens empty, but keep safe.
            ctx = max(1, int(self.decode_kv_lens[0]))
            return [(tokens, ctx)]
        total_kv = max(0, int(self.cached_decode_tokens or 0))
        mean_ctx = max(1, total_kv // n_req) if total_kv else 1
        # One synthetic request carrying all decode tokens at mean ctx approximates
        # sum_i t_i * ctx_i when all ctx equal; uneven cases need the lists.
        return [(tokens, mean_ctx)]


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
