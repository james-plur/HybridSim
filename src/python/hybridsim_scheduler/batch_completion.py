"""Batch completion paths mirroring Frontier ClusterBatchEnd / GlobalBatchEnd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from frontier.entities import Batch
from frontier.entities.batch import Batch as SingleBatch
from frontier.types import ClusterType


@dataclass(frozen=True)
class KVTransferWork:
    transfer_info: Any
    duration_s: float
    workload_id: int


@dataclass
class BatchCompletionPlan:
    reschedule_replica: bool = True
    kv_transfers: list[KVTransferWork] | None = None


def complete_prefill_batch(
    *,
    time_s: float,
    batch: Batch,
    replica_id: int,
    dp_id: int,
    cluster_scheduler,
    kv_cache_transfer_predictor: Any,
    batch_schedule_epoch: int,
    next_workload_id: int,
) -> tuple[BatchCompletionPlan, int]:
    if batch.schedule_epoch != batch_schedule_epoch:
        return BatchCompletionPlan(reschedule_replica=False), next_workload_id

    replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(replica_id, dp_id)
    batch.on_batch_end(time_s, ClusterType.PREFILL)
    replica_scheduler.on_batch_end(batch)

    if kv_cache_transfer_predictor is None:
        raise ValueError("KV cache transfer predictor required for PREFILL completion")

    replica_config = cluster_scheduler._config.replica_config
    target_cluster = cluster_scheduler._get_decode_target_cluster()
    transfers: list[KVTransferWork] = []

    for request in batch.requests:
        if request.is_prefill_complete and request.num_decode_tokens > 0:
            kv_cache_size_bytes, transfer_time_ms = (
                kv_cache_transfer_predictor.get_transfer_info_for_request(
                    source_cluster_type=ClusterType.PREFILL,
                    target_cluster_type=target_cluster,
                    request=request,
                    replica_config=replica_config,
                )
            )
            single_request_batch = SingleBatch(
                replica_id=replica_id,
                requests=[request],
                num_tokens=[request.num_prefill_tokens],
                is_moe=replica_config.model_config.is_moe,
            )
            from frontier.entities.kv_cache_transfer_info import KVCacheTransferInfo

            transfer_info = KVCacheTransferInfo(
                batch=single_request_batch,
                source_cluster_type=ClusterType.PREFILL,
                target_cluster_type=target_cluster,
                source_replica_id=replica_id,
                source_dp_id=dp_id,
                kv_cache_size_bytes=kv_cache_size_bytes,
                transfer_time_ms=transfer_time_ms,
                transfer_start_time=time_s,
            )
            workload_id = next_workload_id
            next_workload_id += 1
            transfers.append(
                KVTransferWork(
                    transfer_info=transfer_info,
                    duration_s=transfer_time_ms * 1e-3,
                    workload_id=workload_id,
                )
            )

    return BatchCompletionPlan(reschedule_replica=True, kv_transfers=transfers), next_workload_id


def complete_global_batch(
    *,
    time_s: float,
    batch: Batch,
    cluster_type: ClusterType,
    replica_id: int,
    dp_id: int,
    cluster_scheduler,
    batch_schedule_epoch: int,
    request_execution_signatures: list[tuple[int, int, int]],
    request_mutation_signatures: list[tuple[int, int, int, int]],
    thinking_round_start_times: list[float | None],
) -> BatchCompletionPlan:
    if batch.schedule_epoch != batch_schedule_epoch:
        return BatchCompletionPlan(reschedule_replica=False)

    replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(replica_id, dp_id)
    batch.on_batch_end(
        time_s,
        cluster_type,
        request_execution_signatures=request_execution_signatures,
        request_mutation_signatures=request_mutation_signatures,
        thinking_round_start_times=thinking_round_start_times,
    )
    replica_scheduler.on_batch_end(batch)
    return BatchCompletionPlan(reschedule_replica=True)


def handle_kv_transfer_complete(
    *,
    time_s: float,
    transfer_info: KVCacheTransferInfo,
    global_scheduler,
    source_cluster_type: ClusterType = ClusterType.PREFILL,
) -> list[ClusterType]:
    transfer_duration_s = time_s - transfer_info.transfer_start_time
    batch = transfer_info.batch
    for request in batch.requests:
        request.on_kv_cache_transfer_complete(time_s, transfer_duration_s)

    target_cluster_scheduler = global_scheduler.get_cluster_scheduler(
        transfer_info.target_cluster_type
    )
    target_cluster_scheduler.on_kv_cache_arrival(time_s, batch, transfer_info)

    source_cluster_scheduler = global_scheduler.get_cluster_scheduler(source_cluster_type)
    source_replica_scheduler = source_cluster_scheduler.get_dp_replica_scheduler(
        transfer_info.source_replica_id,
        transfer_info.source_dp_id,
    )
    source_replica_scheduler.complete_kv_transfer_for_requests(batch.requests)

    clusters_to_schedule: list[ClusterType] = [transfer_info.target_cluster_type]
    if source_replica_scheduler.should_schedule_after_kv_transfer_completion():
        clusters_to_schedule.append(source_cluster_type)
    return clusters_to_schedule
