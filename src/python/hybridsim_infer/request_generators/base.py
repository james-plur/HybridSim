"""Request generators: arrival process → list[InferenceRequest] for ClusterScheduler."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hybridsim_infer.request import InferenceRequest


class RequestGenerator(ABC):
    """Pluggable builder for request arrivals fed to ``schedule_arrivals``.

    Orthogonal to ``InferWorkloadGenerator`` (which builds Engine TimeoutKernels
    from an already-scheduled ``ScheduleBatch``).
    """

    @abstractmethod
    def generate(self) -> list[InferenceRequest]:
        """Return requests; callers should sort by ``arrived_at`` if not already ordered."""
