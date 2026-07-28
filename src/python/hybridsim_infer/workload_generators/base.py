"""Workload generators: schedule / KV transfer → EngineActor workload dict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hybridsim_infer.schedule_types import ScheduleBatch


class WorkloadGenerator(ABC):
    """Pluggable builder for schedule batches: ``workload = generator(batch, workload_id=...)``.

    KV transfer workloads use ``KvTransferWorkloadGenerator`` (sibling API).
    """

    @abstractmethod
    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        """Return an EngineActor workload dict for ``batch``."""
