"""Translate scheduler batches into hybridsim EngineActor workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.entities import Batch
from frontier.execution_time_predictor import BaseExecutionTimePredictor
from frontier.types import ClusterType


@dataclass(frozen=True)
class BatchWorkload:
    workload_id: int
    duration: float
    batch: Batch
    batch_schedule_epoch: int
    request_execution_signatures: list[tuple[int, int, int]]
    request_mutation_signatures: list[tuple[int, int, int, int]]
    thinking_round_start_times: list[float | None]


def predict_batch_duration(
    batch: Batch,
    predictor: BaseExecutionTimePredictor,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
) -> float:
    num_layers = predictor._num_layers_per_pipeline_stage
    execution_time = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=cluster_type,
        num_layers=num_layers,
    )
    return execution_time.total_time


def batch_to_workload_dict(workload_id: int, batch: Batch, duration: float) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "kernels": [
            {
                "name": f"batch_{batch.id}",
                "duration": duration,
                "dependencies": [],
            }
        ],
    }


def build_batch_workload(
    *,
    workload_id: int,
    batch: Batch,
    predictor: BaseExecutionTimePredictor,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
) -> BatchWorkload:
    duration = predict_batch_duration(batch, predictor, cluster_type)
    return BatchWorkload(
        workload_id=workload_id,
        duration=duration,
        batch=batch,
        batch_schedule_epoch=batch.schedule_epoch,
        request_execution_signatures=list(batch.request_execution_signatures),
        request_mutation_signatures=list(batch.request_mutation_signatures),
        thinking_round_start_times=list(batch.thinking_round_start_times),
    )
