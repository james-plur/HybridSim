"""Single TimeoutKernel workload (fake GPU batch execution)."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.schedule_types import ScheduleBatch
from hybridsim_infer.workload_generators.base import WorkloadGenerator
from hybridsim_infer.workload_generators.predictors import (
    BatchDurationPredictor,
    FixedDurationPredictor,
)


class TimeoutKernelWorkloadGenerator(WorkloadGenerator):
    """Map ``ScheduleBatch`` → one TimeoutKernel whose duration comes from a predictor."""

    def __init__(
        self,
        predictor: Optional[BatchDurationPredictor] = None,
    ) -> None:
        self._predictor: BatchDurationPredictor = predictor or FixedDurationPredictor()

    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        duration_s = float(self._predictor.predict(batch))
        return {
            "workload_id": int(workload_id),
            "kernels": [
                {
                    "name": f"batch_{batch.batch_id}",
                    "duration": duration_s,
                    "dependencies": [],
                }
            ],
        }
