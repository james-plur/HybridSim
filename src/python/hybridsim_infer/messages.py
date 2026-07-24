"""Message types for the native inference simulation actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hybridsim_infer.request import InferenceRequest


@dataclass
class RequestArriveMsg:
    request: InferenceRequest


@dataclass
class RequestMsg:
    request: InferenceRequest


@dataclass
class RequestFinishMsg:
    request: InferenceRequest
    replica_id: int = 0


@dataclass
class StepMsg:
    """Replica self-scheduling tick (while-step loop)."""

    pass


@dataclass
class BatchEndMsg:
    workload_id: int
    batch: Any = None


@dataclass
class KVTransferEndMsg:
    request_id: int = 0


@dataclass
class KVLookupMsg:
    token_ids: list[int] = field(default_factory=list)


@dataclass
class KVUpdateMsg:
    token_ids: list[int] = field(default_factory=list)
    request_id: int = 0


INFER_MESSAGE_TYPES: tuple[type, ...] = (
    RequestArriveMsg,
    RequestMsg,
    RequestFinishMsg,
    StepMsg,
    BatchEndMsg,
    KVTransferEndMsg,
    KVLookupMsg,
    KVUpdateMsg,
)
