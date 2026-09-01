"""ScheduleBatch → mock Operator DAG → kernel workload (op-level)."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.schedule_types import ScheduleBatch
from hybridsim_infer.workload_generators.configs import (
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    OpLevelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.infer_workload_generator.base import (
    InferWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    extract_batch_features,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analyzer import (
    OpAnalyzer,
    analyze_split,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    AnalyticAnalyzer,
    critical_path_duration_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    build_operator_dag,
)


class OpLevelWorkloadGenerator(InferWorkloadGenerator):
    """Build Operator DAG via mock ``forward``, then analyze into kernels.

    Compute ops always go through ``compute_analyzer`` (default:
    ``AnalyticAnalyzer``). Communication ops use ``comm_analyzer`` when set
    (e.g. ``RingCommAnalyzer`` → per-rank Put/Wait); otherwise the compute
    analyzer also lowers CommOp (α-β TimeoutKernel).
    """

    def __init__(
        self,
        *,
        op_level: Optional[OpLevelConfig] = None,
        model: Optional[ModelConfig] = None,
        parallel: Optional[ParallelConfig] = None,
        device: Optional[DeviceConfig] = None,
        network: Optional[NetworkConfig] = None,
        analyzer: Optional[OpAnalyzer] = None,
        compute_analyzer: Optional[OpAnalyzer] = None,
        comm_analyzer: Optional[OpAnalyzer] = None,
        comm_parser: Any = None,
        replica_id: int = 0,
        num_ranks: int = 1,
    ) -> None:
        if op_level is not None:
            self._cfg = op_level
        else:
            self._cfg = OpLevelConfig(
                model=model or ModelConfig(),
                parallel=parallel or ParallelConfig(),
                device=device or DeviceConfig(),
                network=network or NetworkConfig(),
            )
        compute = compute_analyzer or analyzer
        self._compute = compute or AnalyticAnalyzer(
            device=self._cfg.device,
            network=self._cfg.network,
            duration_scale=self._cfg.duration_scale,
        )
        self._comm = comm_analyzer if comm_analyzer is not None else comm_parser
        self._replica_id = int(replica_id)
        self._num_ranks = max(1, int(num_ranks))

    @property
    def config(self) -> OpLevelConfig:
        return self._cfg

    @property
    def analyzer(self) -> OpAnalyzer:
        return self._compute

    @property
    def compute_analyzer(self) -> OpAnalyzer:
        return self._compute

    @property
    def comm_analyzer(self) -> Optional[OpAnalyzer]:
        return self._comm

    def build_dag(self, batch: ScheduleBatch):
        features = extract_batch_features(batch)
        return build_operator_dag(
            model=self._cfg.model,
            parallel=self._cfg.parallel,
            batch=features,
        )

    def predict_duration_s(
        self,
        batch: ScheduleBatch,
        *,
        metric: str = "critical_path",
    ) -> float:
        """Op-level duration for ``batch`` (uses config ``duration_scale``).

        Uses only the compute analyzer so CommOp is α-β TimeoutKernel and the
        critical path has finite duration.
        """
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
            total_kernel_duration_s,
        )

        op_dag = self.build_dag(batch)
        wl = self._compute.analyze(op_dag, workload_id=0)
        kernels = wl.get("kernels") or []
        if metric == "sum":
            return total_kernel_duration_s(kernels)
        return critical_path_duration_s(kernels)

    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        op_dag = self.build_dag(batch)
        if self._comm is None:
            return self._compute.analyze(op_dag, workload_id=workload_id)
        per_rank: dict[int, dict[str, Any]] = {}
        for rank in range(self._num_ranks):
            wl = analyze_split(
                op_dag,
                compute=self._compute,
                comm=self._comm,
                workload_id=workload_id,
                rank=rank,
                replica_id=self._replica_id,
                num_ranks=self._num_ranks,
            )
            per_rank[int(rank)] = {"kernels": wl["kernels"]}
        return {
            "workload_id": int(workload_id),
            "per_rank": per_rank,
        }
