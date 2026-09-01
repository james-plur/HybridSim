"""Analytic analyzer: Roofline / α-β TimeoutKernels from an OperatorDAG."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.configs import DeviceConfig, NetworkConfig
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analyzer import (
    AnalyzeContext,
    OpAnalyzer,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.lower import (
    lower_op,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.models.ab_comm import (
    ab_comm_time_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.models.roofline import (
    mem_time_s,
    roofline_time_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    KernelPlan,
    OperatorDAG,
    OperatorKind,
)


def critical_path_duration_s(kernels: list[dict[str, Any]]) -> float:
    """Longest path duration through a TimeoutKernel DAG."""
    n = len(kernels)
    if n == 0:
        return 0.0
    dist = [0.0] * n
    for i, k in enumerate(kernels):
        deps = k.get("dependencies") or []
        pred = max((dist[d] for d in deps), default=0.0)
        dist[i] = pred + float(k.get("duration", 0.0))
    return max(dist) if dist else 0.0


def total_kernel_duration_s(kernels: list[dict[str, Any]]) -> float:
    return sum(float(k.get("duration", 0.0)) for k in kernels)


class AnalyticAnalyzer(OpAnalyzer):
    """Estimate operator durations (Roofline / α-β) and emit TimeoutKernels.

    Handles compute ops always. Also lowers ``CommOp`` via α-β when no separate
    comm analyzer is configured.

    ``duration_scale`` is a static knob (typically filled after offline
    calibration); this module does not run RF fitting.
    """

    def __init__(
        self,
        device: DeviceConfig | None = None,
        network: NetworkConfig | None = None,
        *,
        duration_scale: float = 1.0,
        mem_scale: float = 1.0,
    ) -> None:
        self.device = device or DeviceConfig()
        self.network = network or NetworkConfig()
        self.duration_scale = float(duration_scale)
        self.mem_scale = float(mem_scale)

    def handles(self, op: Any) -> bool:
        return True

    def estimate_kernel_duration(self, plan: KernelPlan) -> float:
        feats = plan.features or {}
        if plan.kind is OperatorKind.COMM:
            raw = ab_comm_time_s(
                payload_bytes=float(feats.get("payload_bytes", 0.0)),
                volume_factor=float(feats.get("volume_factor", 0.0)),
                network=self.network,
                num_ranks=int(feats.get("num_ranks", 1)),
            )
        elif plan.kind is OperatorKind.MEM:
            raw = mem_time_s(
                bytes_=float(feats.get("bytes", 0.0)),
                device=self.device,
                mem_scale=self.mem_scale,
            )
        else:
            raw = roofline_time_s(
                flops=float(feats.get("flops", 0.0)),
                bytes_=float(feats.get("bytes", 0.0)),
                device=self.device,
            )
        return float(raw) * self.duration_scale

    def lower_op(
        self,
        op: Any,
        *,
        op_index: int,
        ctx: AnalyzeContext,
    ) -> list[dict[str, Any]]:
        _ = (op_index, ctx)
        plan = lower_op(op)
        return [
            {
                "name": plan.name,
                "duration": float(self.estimate_kernel_duration(plan)),
                "rel_deps": [],
            }
        ]

    def analyze(
        self,
        op_dag: OperatorDAG,
        *,
        workload_id: int,
        rank: int | None = None,
        replica_id: int = 0,
        num_ranks: int = 1,
    ) -> dict[str, Any]:
        """Lower every op (compute and comm) to TimeoutKernels."""
        return super().analyze(
            op_dag,
            workload_id=workload_id,
            rank=rank,
            replica_id=replica_id,
            num_ranks=num_ranks,
        )
