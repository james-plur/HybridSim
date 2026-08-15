"""Inference scheduler ABC + name registry (vLLM / future SGLang / …)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, Type, Union

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import ScheduleResult

# Sync or async remote lookup: (request) -> {"hit": bool, "num_tokens": int}
RemoteLookupFn = Callable[
    [InferenceRequest],
    Union[dict[str, Any], Awaitable[dict[str, Any]]],
]


class InferenceScheduler(ABC):
    """Replica-local schedule + batch-completion policy for one serving stack."""

    name: str = "base"

    @abstractmethod
    async def schedule_step(
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


class SchedulerFactory:
    """Simple name → class registry for replica schedulers / offline drivers."""

    _registry: dict[str, Type[InferenceScheduler]] = {}

    @classmethod
    def register(cls, name: str, scheduler_cls: Type[InferenceScheduler]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("scheduler name must be non-empty")
        cls._registry[key] = scheduler_cls

    @classmethod
    def create(cls, name: str = "vllm", **kwargs: Any) -> InferenceScheduler:
        key = (name or "vllm").strip().lower()
        scheduler_cls = cls._registry.get(key)
        if scheduler_cls is None:
            known = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(f"unknown scheduler {name!r}; registered: {known}")
        return scheduler_cls(**kwargs)

    @classmethod
    def registered(cls) -> list[str]:
        return sorted(cls._registry)
