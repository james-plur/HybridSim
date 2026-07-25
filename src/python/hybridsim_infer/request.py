"""Inference request entity (lightweight, vLLM-inspired fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


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
    #: Sampled output tokens so far (survives preemption; used for full-ISL admit).
    num_output_tokens: int = 0
    #: Prompt token ids used for local/remote prefix match. Empty → synthetic ids.
    prompt_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    completed: bool = False
    #: Tokens expected from an in-flight remote KV pull.
    pending_remote_tokens: int = 0
    #: Mooncake-style transfer params (``do_remote_prefill`` / ``do_remote_decode`` / …).
    kv_transfer_params: Optional[dict] = None
    #: Store async lookup in flight (request stays WAITING).
    pending_lookup: bool = False
    #: Cached async lookup result; consumed by the next ``remote_lookup``.
    lookup_result: Optional[dict] = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids and self.num_prefill_tokens > 0:
            # Stable synthetic prompt for prefix-cache demos.
            self.prompt_token_ids = [
                (self.request_id * 1_000_003 + i) % 50_000
                for i in range(self.num_prefill_tokens)
            ]

    @property
    def num_tokens(self) -> int:
        """Current sequence length (prompt + outputs), matching vLLM ``Request.num_tokens``."""
        return self.num_prefill_tokens + self.num_output_tokens

    @property
    def num_tokens_with_output(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.num_tokens_with_output - self.num_computed_tokens)

    def is_finished(self) -> bool:
        return self.num_computed_tokens >= self.num_tokens_with_output

    @property
    def is_prefill_chunk(self) -> bool:
        return self.num_computed_tokens < self.num_prefill_tokens
