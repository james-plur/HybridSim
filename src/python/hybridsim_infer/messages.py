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
class RequestHandoffMsg:
    """Prefill done → Cluster routes the request to a Decode replica (P2P)."""

    request: InferenceRequest
    from_replica_id: int = 0
    transfer_id: str = ""


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
    #: ``pull`` clears WAIT_FOR_REMOTE_KVS; ``push`` is fire-and-forget for schedule.
    direction: str = "pull"


@dataclass
class KVLookupMsg:
    token_ids: list[int] = field(default_factory=list)
    block_keys: list[str] = field(default_factory=list)
    request_id: int = 0
    async_reply: bool = False
    reply_to: Any = None


@dataclass
class KVLookupReplyMsg:
    """Async lookup result delivered to the requesting replica (cache only)."""

    request_id: int = 0
    hit: bool = False
    num_tokens: int = 0
    num_blocks: int = 0
    location: Any = None


@dataclass
class KVUpdateMsg:
    token_ids: list[int] = field(default_factory=list)
    block_keys: list[str] = field(default_factory=list)
    request_id: int = 0


INFER_MESSAGE_TYPES: tuple[type, ...] = (
    RequestArriveMsg,
    RequestMsg,
    RequestFinishMsg,
    RequestHandoffMsg,
    StepMsg,
    BatchEndMsg,
    KVTransferEndMsg,
    KVLookupMsg,
    KVLookupReplyMsg,
    KVUpdateMsg,
)
