"""Infer workload generators: ScheduleBatch → EngineActor workload dict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hybridsim_infer.schedule_types import ScheduleBatch


class InferWorkloadGenerator(ABC):
    """Pluggable builder for schedule batches: ``workload = generator(batch, workload_id=...)``."""

    @abstractmethod
    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        """Return an EngineActor workload dict for ``batch``."""
