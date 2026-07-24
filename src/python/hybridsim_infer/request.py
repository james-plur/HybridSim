"""Inference request entity (lightweight, vLLM-inspired fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    WAIT_FOR_REMOTE_KVS = auto()
    PREEMPTED = auto()
    FINISHED = auto()


@dataclass
class InferenceRequest:
    request_id: int
    arrived_at: float = 0.0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    completed: bool = False

    @property
    def num_tokens_with_output(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.num_tokens_with_output - self.num_computed_tokens)

    def is_finished(self) -> bool:
        return self.num_computed_tokens >= self.num_tokens_with_output
