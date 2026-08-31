"""ScheduleBatch → mock Operator DAG → TimeoutKernel workload (op-level)."""

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
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    AnalyticAnalyzer,
    critical_path_duration_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    build_operator_dag,
)


class OpLevelWorkloadGenerator(InferWorkloadGenerator):
    """Build Operator DAG via mock ``forward``, then analyze into TimeoutKernels."""

    def __init__(
        self,
        *,
        op_level: Optional[OpLevelConfig] = None,
        model: Optional[ModelConfig] = None,
        parallel: Optional[ParallelConfig] = None,
        device: Optional[DeviceConfig] = None,
        network: Optional[NetworkConfig] = None,
        analyzer: Optional[AnalyticAnalyzer] = None,
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
        self._analyzer = analyzer or AnalyticAnalyzer(
            device=self._cfg.device,
            network=self._cfg.network,
            duration_scale=self._cfg.duration_scale,
        )

    @property
    def config(self) -> OpLevelConfig:
        return self._cfg

    @property
    def analyzer(self) -> AnalyticAnalyzer:
        return self._analyzer

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
        """Op-level duration for ``batch`` (uses config ``duration_scale``)."""
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
            total_kernel_duration_s,
        )

        wl = self(batch, workload_id=0)
        if metric == "sum":
            return total_kernel_duration_s(wl["kernels"])
        return critical_path_duration_s(wl["kernels"])

    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        op_dag = self.build_dag(batch)
        return self._analyzer.analyze(op_dag, workload_id=workload_id)
