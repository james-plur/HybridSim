"""Shared schedule DTOs, cluster dispatch, and workload helpers.

vLLM Phase1/2 schedule lives in ``hybridsim_infer.frameworks.VllmFramework``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hybridsim_infer.request import InferenceRequest


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
    """Minimal batch handed to the worker engine (SchedulerOutput-ish)."""

    batch_id: int
    chunks: list[Any] = field(default_factory=list)
    requests: list[InferenceRequest] = field(default_factory=list)
    tokens_per_request: dict[int, int] = field(default_factory=dict)
    req_to_new_blocks: dict[int, list[Any]] = field(default_factory=dict)


@dataclass
class RemoteKvPull:
    request: InferenceRequest
    num_tokens: int
    token_ids: list[int]


@dataclass
class ScheduleResult:
    waiting: list[InferenceRequest]
    running: list[InferenceRequest]
    batch: Optional[ScheduleBatch]
    remote_pulls: list[RemoteKvPull] = field(default_factory=list)
    preempted: list[InferenceRequest] = field(default_factory=list)
    finished_cached: list[InferenceRequest] = field(default_factory=list)
    #: If True, Phase 2 stopped early after queuing a remote KV pull.
    stop_after_remote: bool = False


def dispatch(
    request: InferenceRequest,
    replica_loads: list[int],
) -> int:
    """Pick a replica id for ``request``. Stub: least-loaded."""
    if not replica_loads:
        return 0
    return min(range(len(replica_loads)), key=lambda i: replica_loads[i])


def inference_workload_generator(
    schedule_batch: ScheduleBatch,
    *,
    workload_id: int,
    duration_s: float,
) -> dict[str, Any]:
    """Turn a schedule batch into an EngineActor workload (single TimeoutKernel)."""
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


def kv_transfer_workload(
    *,
    workload_id: int,
    request_id: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "kernels": [
            {
                "name": f"kv_xfer_{request_id}",
                "duration": duration_s,
                "dependencies": [],
            }
        ],
    }
