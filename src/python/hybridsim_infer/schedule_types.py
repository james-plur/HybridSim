"""Schedule DTOs and cluster dispatch helpers.

Used by frameworks, actors, and workload_generators.
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
