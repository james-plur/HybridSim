"""Inference framework abstraction (vLLM / future SGLang / …)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.stubs import ScheduleResult

# Sync remote lookup: (request) -> {"hit": bool, "num_tokens": int}
RemoteLookupFn = Callable[[InferenceRequest], dict[str, Any]]


class InferenceFramework(ABC):
    """Replica-local schedule + batch-completion policy for one serving stack."""

    name: str = "base"

    @abstractmethod
    def schedule_step(
        self,
        waiting: list[InferenceRequest],
        running: list[InferenceRequest],
        *,
        kv_cache_manager: Any,
        batch_id: int,
        token_budget: int,
        max_num_running_reqs: int,
        remote_lookup: Optional[RemoteLookupFn] = None,
    ) -> ScheduleResult:
        """One schedule tick: update queues and optionally emit a batch."""

    @abstractmethod
    def on_batch_complete(
        self,
        requests: list[InferenceRequest],
        tokens_per_request: dict[int, int],
        kv_cache_manager: Any,
    ) -> list[InferenceRequest]:
        """Advance computed/output after a fake/real forward; return newly finished."""
