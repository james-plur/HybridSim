"""Operator base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchPhase,
    KernelPlan,
    OperatorKind,
)


@dataclass
class Operator(ABC):
    """Python-level operator node in an OperatorDAG.

    Duration is assigned later by OpAnalyzer after ``expand_kernels``.
    """

    name: str
    kind: OperatorKind
    variant: str
    phase: BatchPhase
    deps: list[int] = field(default_factory=list)
    #: Shared workload features (tokens, dims, payload, ranks, ...).
    features: dict[str, Any] = field(default_factory=dict)
    layer_id: int = 0

    @abstractmethod
    def expand_kernels(self) -> list[KernelPlan]:
        """Expand this operator into one or more TimeoutKernel plans."""

    def _single_kernel(self, *, extra: dict[str, Any] | None = None) -> list[KernelPlan]:
        feats = dict(self.features)
        if extra:
            feats.update(extra)
        return [
            KernelPlan(
                name=self.name,
                local_deps=[],
                features=feats,
                kind=self.kind,
            )
        ]
