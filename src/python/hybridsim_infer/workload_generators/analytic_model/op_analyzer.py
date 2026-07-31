"""Analyze OperatorDAG → Engine TimeoutKernel workload dict."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.workload_generators.analytic_model.configs import (
    DeviceConfig,
    NetworkConfig,
)
from hybridsim_infer.workload_generators.analytic_model.models.ab_comm import (
    ab_comm_time_s,
)
from hybridsim_infer.workload_generators.analytic_model.models.roofline import (
    roofline_time_s,
)
from hybridsim_infer.workload_generators.analytic_model.operators.base import Operator
from hybridsim_infer.workload_generators.analytic_model.types import (
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


class OpAnalyzer:
    """Estimate Operator durations (Roofline / α-β) and emit TimeoutKernels.

    ``duration_scale`` is a static knob (typically filled after offline
    calibration); this module does not run RF fitting.
    """

    def __init__(
        self,
        device: DeviceConfig | None = None,
        network: NetworkConfig | None = None,
        *,
        duration_scale: float = 1.0,
    ) -> None:
        self.device = device or DeviceConfig()
        self.network = network or NetworkConfig()
        self.duration_scale = float(duration_scale)

    def estimate_kernel_duration(self, plan: KernelPlan) -> float:
        feats = plan.features or {}
        if plan.kind is OperatorKind.COMM:
            raw = ab_comm_time_s(
                payload_bytes=float(feats.get("payload_bytes", 0.0)),
                volume_factor=float(feats.get("volume_factor", 0.0)),
                network=self.network,
                num_ranks=int(feats.get("num_ranks", 1)),
            )
        else:
            raw = roofline_time_s(
                flops=float(feats.get("flops", 0.0)),
                bytes_=float(feats.get("bytes", 0.0)),
                device=self.device,
            )
        return float(raw) * self.duration_scale

    def analyze(
        self,
        op_dag: OperatorDAG,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        """Expand operators and remap dependencies into an Engine workload dict."""
        kernels: list[dict[str, Any]] = []
        op_to_kernels: list[list[int]] = []

        for op_idx, op in enumerate(op_dag.operators):
            if not isinstance(op, Operator):
                raise TypeError(f"DAG node {op_idx} is not an Operator: {type(op)!r}")
            plans = op.expand_kernels()
            if not plans:
                raise ValueError(f"Operator {op.name!r} expanded to zero kernels")

            cross_deps: list[int] = []
            for dep_op in op.deps:
                if dep_op < 0 or dep_op >= op_idx:
                    raise ValueError(
                        f"Operator {op.name!r} has invalid dep {dep_op} "
                        f"(must be earlier operator index)"
                    )
                pred_kernels = op_to_kernels[dep_op]
                if not pred_kernels:
                    raise ValueError(f"Predecessor op {dep_op} produced no kernels")
                cross_deps.append(pred_kernels[-1])

            local_base = len(kernels)
            produced: list[int] = []
            for local_i, plan in enumerate(plans):
                deps: list[int] = []
                if local_i == 0:
                    deps.extend(cross_deps)
                for ld in plan.local_deps:
                    if ld < 0 or ld >= local_i:
                        raise ValueError(
                            f"Kernel {plan.name!r} has invalid local_dep {ld}"
                        )
                    deps.append(local_base + ld)
                seen: set[int] = set()
                uniq_deps: list[int] = []
                for d in deps:
                    if d not in seen:
                        seen.add(d)
                        uniq_deps.append(d)

                duration = self.estimate_kernel_duration(plan)
                kid = len(kernels)
                kernels.append(
                    {
                        "name": plan.name,
                        "duration": float(duration),
                        "dependencies": uniq_deps,
                    }
                )
                produced.append(kid)
            op_to_kernels.append(produced)

        return {
            "workload_id": int(workload_id),
            "kernels": kernels,
        }
