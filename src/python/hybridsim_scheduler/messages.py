"""Message types for hybridsim scheduler actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RequestArrivalMsg:
    request: Any


@dataclass
class ClusterScheduleMsg:
    cluster_type: Any


@dataclass
class ReplicaScheduleMsg:
    cluster_type: Any


@dataclass
class BatchCompleteMsg:
    workload_id: int
    cluster_type: Any
    batch_schedule_epoch: int
    request_execution_signatures: list[tuple[int, int, int]]
    request_mutation_signatures: list[tuple[int, int, int, int]]
    thinking_round_start_times: list[float | None]


@dataclass
class KVTransferCompleteMsg:
    transfer_info: Any


def register_scheduler_messages(sim) -> dict[str, Any]:
    """Register scheduler message types on a hybridsim Simulation."""
    return {
        "RequestArrivalMsg": sim.register_message(RequestArrivalMsg),
        "ClusterScheduleMsg": sim.register_message(ClusterScheduleMsg),
        "ReplicaScheduleMsg": sim.register_message(ReplicaScheduleMsg),
        "BatchCompleteMsg": sim.register_message(BatchCompleteMsg),
        "KVTransferCompleteMsg": sim.register_message(KVTransferCompleteMsg),
    }
