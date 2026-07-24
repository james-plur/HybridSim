"""Extensible scheduling hooks; dummy path enables end-to-end smoke tests.

Replace bodies with vLLM/SGLang-aligned logic later (see design doc +
vLLM Engine schedule notes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hybridsim_infer.request import InferenceRequest, RequestStatus


@dataclass
class PrefillChunk:
    request: InferenceRequest
    num_tokens: int


@dataclass
class DecodeChunk:
    request: InferenceRequest
    num_tokens: int = 1


@dataclass
class ScheduleBatch:
    """Minimal batch handed to the worker engine."""

    batch_id: int
    chunks: list[Any] = field(default_factory=list)
    requests: list[InferenceRequest] = field(default_factory=list)
    tokens_per_request: dict[int, int] = field(default_factory=dict)


def dispatch(
    request: InferenceRequest,
    replica_loads: list[int],
) -> int:
    """Pick a replica id for ``request``. Stub: round-robin by load."""
    # TODO: PD-aware / load-aware dispatch
    if not replica_loads:
        return 0
    return min(range(len(replica_loads)), key=lambda i: replica_loads[i])


def process_wait_queue(
    waiting: list[InferenceRequest],
    *,
    kv_cache_manager: Any,
    tokens_per_step: int,
) -> tuple[list[InferenceRequest], list[InferenceRequest], list[PrefillChunk]]:
    """Admit waiting requests into running / produce prefill chunks (dummy).

    Returns ``(still_waiting, newly_admitted, prefill_chunks)``.
    """
    # TODO: WAIT_FOR_REMOTE_KVS, prefix match, remote KV lookup, chunked prefill
    still_waiting: list[InferenceRequest] = []
    newly_admitted: list[InferenceRequest] = []
    prefill_chunks: list[PrefillChunk] = []
    for req in waiting:
        if req.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
            still_waiting.append(req)
            continue
        if req.status != RequestStatus.WAITING:
            still_waiting.append(req)
            continue
        remaining_prefill = max(0, req.num_prefill_tokens - req.num_computed_tokens)
        if remaining_prefill > 0:
            n = min(tokens_per_step, remaining_prefill)
            blocks = kv_cache_manager.allocate(req, n)
            if blocks is None:
                still_waiting.append(req)
                continue
            prefill_chunks.append(PrefillChunk(request=req, num_tokens=n))
            req.status = RequestStatus.RUNNING
            newly_admitted.append(req)
        else:
            req.status = RequestStatus.RUNNING
            newly_admitted.append(req)
    return still_waiting, newly_admitted, prefill_chunks


def process_running_queue(
    running: list[InferenceRequest],
    *,
    kv_cache_manager: Any,
    tokens_per_step: int,
) -> tuple[list[InferenceRequest], list[PrefillChunk], list[DecodeChunk]]:
    """Continue chunked prefill and/or schedule decode for running requests (dummy).

    Returns ``(still_running, prefill_chunks, decode_chunks)``.
    """
    # TODO: preemption when allocate fails; align with vLLM running-queue policy
    still_running: list[InferenceRequest] = []
    prefill_chunks: list[PrefillChunk] = []
    decode_chunks: list[DecodeChunk] = []
    for req in running:
        if req.is_finished():
            continue
        remaining_prefill = max(0, req.num_prefill_tokens - req.num_computed_tokens)
        if remaining_prefill > 0:
            n = min(tokens_per_step, remaining_prefill)
            blocks = kv_cache_manager.allocate(req, n)
            if blocks is None:
                still_running.append(req)
                continue
            prefill_chunks.append(PrefillChunk(request=req, num_tokens=n))
            still_running.append(req)
            continue
        n = min(tokens_per_step, req.remaining_tokens)
        if n <= 0:
            continue
        blocks = kv_cache_manager.allocate(req, n)
        if blocks is None:
            still_running.append(req)
            continue
        decode_chunks.append(DecodeChunk(request=req, num_tokens=n))
        still_running.append(req)
    return still_running, prefill_chunks, decode_chunks


def batch(
    prefill_chunks: list[PrefillChunk],
    decode_chunks: list[DecodeChunk],
    *,
    batch_id: int,
) -> Optional[ScheduleBatch]:
    """Merge prefill/decode chunks into one schedule batch."""
    # TODO: align with vLLM SchedulerOutput → ExecuteModelRequest
    if not prefill_chunks and not decode_chunks:
        return None
    requests: list[InferenceRequest] = []
    tokens: dict[int, int] = {}
    chunks: list[Any] = []
    for c in prefill_chunks:
        chunks.append(c)
        if c.request not in requests:
            requests.append(c.request)
        tokens[c.request.request_id] = tokens.get(c.request.request_id, 0) + c.num_tokens
    for c in decode_chunks:
        chunks.append(c)
        if c.request not in requests:
            requests.append(c.request)
        tokens[c.request.request_id] = tokens.get(c.request.request_id, 0) + c.num_tokens
    return ScheduleBatch(
        batch_id=batch_id,
        chunks=chunks,
        requests=requests,
        tokens_per_request=tokens,
    )


def inference_workload_generator(
    schedule_batch: ScheduleBatch,
    *,
    workload_id: int,
    duration_s: float,
) -> dict[str, Any]:
    """Turn a schedule batch into an EngineActor workload (single TimeoutKernel)."""
    # TODO: expand to multi-kernel DAG via Frontier-style estimator
    return {
        "workload_id": workload_id,
        "kernels": [
            {
                "name": f"batch_{schedule_batch.batch_id}",
                "duration": duration_s,
                "dependencies": [],
            }
        ],
    }
