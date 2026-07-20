"""Build Frontier scheduler objects from SimulationConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from frontier.config import SimulationConfig, get_quantization_manager, global_vars
from frontier.entities import Cluster
from frontier.execution_time_predictor import ExecutionTimePredictorRegistry
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.request_generator import RequestGeneratorRegistry
from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from frontier.types import ClusterType


@dataclass
class SchedulerBundle:
    config: SimulationConfig
    clusters: dict[ClusterType, Cluster]
    predictors: dict[ClusterType, Any]
    global_scheduler: BaseGlobalScheduler
    request_generator: Any
    kv_cache_transfer_predictor: Any = None

    @property
    def is_disaggregated(self) -> bool:
        return self.config.is_disaggregated_mode()

    def cluster_scheduler(self, cluster_type: ClusterType):
        return self.global_scheduler.get_cluster_scheduler(cluster_type)

    def predictor(self, cluster_type: ClusterType):
        return self.predictors[cluster_type]

    def replica_scheduler_keys(self, cluster_type: ClusterType) -> list[tuple[int, int]]:
        return list(self.cluster_scheduler(cluster_type)._dp_replica_schedulers.keys())

    def get_replica_scheduler(self, cluster_type: ClusterType, replica_id: int, dp_id: int):
        return self.cluster_scheduler(cluster_type).get_dp_replica_scheduler(
            replica_id, dp_id
        )


def build_scheduler_bundle(config: SimulationConfig) -> SchedulerBundle:
    """Mirror Frontier Simulator scheduler initialization without DES."""
    cluster_configs = config.get_clusters()
    model_configs = {
        cluster_config.replica_config.model_config.get_name(): cluster_config.replica_config.model_config
        for cluster_config in cluster_configs.values()
    }
    if len(model_configs) != 1:
        raise ValueError(
            "All clusters must share the same model config. "
            f"Found: {sorted(model_configs.keys())}"
        )
    model_config = next(iter(model_configs.values()))

    quantization_manager = get_quantization_manager()
    quantization_manager.configure_from_model_config(model_config)
    global_vars.set_quantization_manager(quantization_manager)

    clusters: dict[ClusterType, Cluster] = {}
    for cluster_type, cluster_config in cluster_configs.items():
        clusters[cluster_type] = Cluster(
            cluster_config,
            config.metrics_config,
            config.request_generator_config,
        )

    predictors: dict[ClusterType, Any] = {}
    model_manager: Optional[ExecutionTimePredictionModelManager] = None

    if config.is_disaggregated_mode():
        model_manager = ExecutionTimePredictionModelManager(
            cluster_configs, config.metrics_config
        )
        for cluster_type, cluster_config in cluster_configs.items():
            cluster = clusters[cluster_type]
            predictors[cluster_type] = ExecutionTimePredictorRegistry.get(
                cluster_config.execution_time_predictor_config.get_type(),
                predictor_config=cluster_config.execution_time_predictor_config,
                replica_config=cluster_config.replica_config,
                replica_scheduler_config=cluster_config.replica_scheduler_config,
                metrics_config=config.metrics_config,
                cluster_config=config.cluster_config,
                model_manager=model_manager,
                cluster_type=cluster_type,
                training_file_paths=model_manager.get_training_file_paths(cluster_type),
                actual_replica_ids=list(cluster.replicas.keys()),
                cc_backend=cluster.cc_backend,
            )
    else:
        cluster_config = cluster_configs[ClusterType.MONOLITHIC]
        cluster = clusters[ClusterType.MONOLITHIC]
        predictors[ClusterType.MONOLITHIC] = ExecutionTimePredictorRegistry.get(
            cluster_config.execution_time_predictor_config.get_type(),
            predictor_config=cluster_config.execution_time_predictor_config,
            replica_config=cluster_config.replica_config,
            replica_scheduler_config=cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
            cluster_config=config.cluster_config,
            model_manager=None,
            cluster_type=ClusterType.MONOLITHIC,
            cc_backend=cluster.cc_backend,
        )

    kv_cache_transfer_predictor = None
    if config.is_disaggregated_mode():
        from frontier.kv_cache_transfer import KVCacheTransferPredictorRegistry

        kv_cache_transfer_predictor = KVCacheTransferPredictorRegistry.get(
            config.kv_cache_transfer_config.get_type(),
            config=config.kv_cache_transfer_config,
        )

    m2n_transfer_predictor = None
    if config.is_disaggregated_mode():
        from frontier.m2n_transfer import M2NTransferPredictorRegistry

        m2n_transfer_predictor = M2NTransferPredictorRegistry.get(
            config.m2n_transfer_config.get_type(),
            config=config.m2n_transfer_config,
        )

    global_scheduler = BaseGlobalScheduler(
        clusters,
        config.request_generator_config,
        predictors=predictors,
        kv_cache_transfer_predictor=kv_cache_transfer_predictor,
        m2n_transfer_predictor=m2n_transfer_predictor,
        enable_parallel_mode=False,
        max_inter_cluster_queue_size=config.max_inter_cluster_queue_size,
    )

    request_generator = RequestGeneratorRegistry.get(
        config.request_generator_config.get_type(),
        config.request_generator_config,
    )
    request_generator.configure_thinking_mode(
        enable_thinking_mode=config.enable_thinking_mode,
        thinking_depth=config.thinking_depth,
        tool_call_latency=config.tool_call_latency,
        thinking_round_prefill_tokens=config.thinking_round_prefill_tokens,
        thinking_round_decode_tokens=config.thinking_round_decode_tokens,
    )

    return SchedulerBundle(
        config=config,
        clusters=clusters,
        predictors=predictors,
        global_scheduler=global_scheduler,
        request_generator=request_generator,
        kv_cache_transfer_predictor=kv_cache_transfer_predictor,
    )
