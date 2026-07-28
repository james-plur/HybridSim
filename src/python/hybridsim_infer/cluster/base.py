"""Cluster dispatch managers (monolith / PD pools)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from hybridsim_infer.request import InferenceRequest


class ClusterManager(ABC):
    """Select replica + stamp request fields; track per-replica load."""

    def __init__(self) -> None:
        self._replicas: list[Any] = []
        self._loads: list[int] = []
        self._next_transfer_id = 1

    def bind_replicas(self, replicas: Sequence[Any]) -> None:
        self._replicas = list(replicas)
        self._loads = [0] * len(self._replicas)

    def _alloc_transfer_id(self) -> str:
        tid = f"xfer-{self._next_transfer_id}"
        self._next_transfer_id += 1
        return tid

    def _least_loaded(self, candidate_ids: Sequence[int]) -> int:
        if not candidate_ids:
            raise RuntimeError("empty replica candidate pool")
        return min((int(i) for i in candidate_ids), key=lambda i: self._loads[i])

    def replica(self, replica_id: int) -> Any:
        return self._replicas[int(replica_id)]

    def on_finish(self, replica_id: int) -> None:
        rid = int(replica_id)
        if 0 <= rid < len(self._loads):
            self._loads[rid] = max(0, self._loads[rid] - 1)

    @abstractmethod
    def on_arrive(self, request: InferenceRequest) -> int:
        """Stamp request and return replica id to send RequestMsg."""

    @abstractmethod
    def on_handoff(
        self,
        request: InferenceRequest,
        *,
        from_replica_id: int,
        transfer_id: str = "",
    ) -> int:
        """Handle Prefill→Decode handoff; return decode replica id."""
