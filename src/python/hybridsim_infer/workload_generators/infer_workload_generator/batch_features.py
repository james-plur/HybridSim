"""Extract BatchFeatures from a ScheduleBatch (shared by op-level costing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch


class BatchPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


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
        n_req = max(1, int(self.batch_size) if self.phase is BatchPhase.DECODE else 1)
        if self.decode_kv_lens:
            ctx = max(1, int(self.decode_kv_lens[0]))
            return [(tokens, ctx)]
        total_kv = max(0, int(self.cached_decode_tokens or 0))
        mean_ctx = max(1, total_kv // n_req) if total_kv else 1
        return [(tokens, mean_ctx)]


def extract_batch_features(batch: ScheduleBatch) -> BatchFeatures:
    """Summarize phase and per-request token / KV lengths from a ScheduleBatch.

    Prefill and decode keep **separate** per-request lists (varlen / no pad):
    - ``prefill_chunk_lens[i]``, ``prefill_cached_lens[i]``
    - ``decode_token_lens[i]``, ``decode_kv_lens[i]``

    Scalar ``cached_prefix_tokens`` / ``cached_decode_tokens`` are sums of the
    prefill-only / decode-only lists respectively (not mixed together).
    """
    prefill_chunk_lens: list[int] = []
    prefill_cached_lens: list[int] = []
    decode_token_lens: list[int] = []
    decode_kv_lens: list[int] = []
    req_ids: set[int] = set()

    def _add_prefill(req: Any, n: int) -> None:
        chunk = max(0, int(n))
        if chunk <= 0:
            return
        hit = int(getattr(req, "num_computed_tokens", 0) or 0) if req is not None else 0
        prefill_chunk_lens.append(chunk)
        prefill_cached_lens.append(max(0, hit))
        if req is not None:
            req_ids.add(int(req.request_id))

    def _add_decode(req: Any, n: int) -> None:
        tokens = max(1, int(n))
        if req is None:
            decode_token_lens.append(tokens)
            decode_kv_lens.append(1)
            return
        computed = int(getattr(req, "num_computed_tokens", 0) or 0)
        prompt = int(getattr(req, "num_prefill_tokens", 0) or 0)
        decode_token_lens.append(tokens)
        decode_kv_lens.append(max(1, max(computed, prompt)))
        req_ids.add(int(req.request_id))

    chunks = list(batch.chunks or [])
    if chunks:
        for ch in chunks:
            if isinstance(ch, PrefillChunk):
                _add_prefill(ch.request, ch.num_tokens)
            elif isinstance(ch, DecodeChunk):
                _add_decode(ch.request, getattr(ch, "num_tokens", 1) or 1)
            else:
                n = int(getattr(ch, "num_tokens", 0) or 0)
                req = getattr(ch, "request", None)
                if req is not None and not bool(
                    getattr(req, "num_computed_tokens", 0)
                    >= getattr(req, "num_prefill_tokens", 0)
                ):
                    _add_prefill(req, n)
                else:
                    _add_decode(req, n or 1)
    else:
        for req in batch.requests or []:
            n = int((batch.tokens_per_request or {}).get(req.request_id, 0))
            computed = int(getattr(req, "num_computed_tokens", 0) or 0)
            prompt = int(getattr(req, "num_prefill_tokens", 0) or 0)
            if computed < prompt:
                _add_prefill(req, n or max(0, prompt - computed))
            else:
                _add_decode(req, n or 1)

    prefill_tokens = sum(prefill_chunk_lens)
    decode_tokens = sum(decode_token_lens)
    cached_prefix_tokens = sum(prefill_cached_lens)
    cached_decode_tokens = sum(decode_kv_lens)

    if prefill_tokens > 0 and decode_tokens > 0:
        phase = BatchPhase.MIXED
    elif decode_tokens > 0:
        phase = BatchPhase.DECODE
    else:
        phase = BatchPhase.PREFILL

    return BatchFeatures(
        phase=phase,
        num_tokens=max(0, prefill_tokens + decode_tokens),
        num_prefill_tokens=max(0, prefill_tokens),
        num_decode_tokens=max(0, decode_tokens),
        batch_size=max(1, len(req_ids) or len(batch.requests or [])),
        cached_decode_tokens=max(0, cached_decode_tokens),
        cached_prefix_tokens=max(0, cached_prefix_tokens),
        prefill_chunk_lens=prefill_chunk_lens,
        prefill_cached_lens=prefill_cached_lens,
        decode_token_lens=decode_token_lens,
        decode_kv_lens=decode_kv_lens,
    )
