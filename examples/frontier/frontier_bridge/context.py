"""Build MONOLITHIC Frontier cluster / replica scheduler context for hybridsim."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    RoundRobinClusterSchedulerConfig,
    SglangSchedulerConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Cluster
from frontier.execution_time_predictor import (
    BaseExecutionTimePredictor,
    ExecutionTimePredictorRegistry,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.cluster_scheduler import ClusterSchedulerRegistry
from frontier.types import ClusterType


class ReplicaSchedulerKind(str, Enum):
    VLLM_V1 = "vllm_v1"
    SGLANG = "sglang"


@dataclass
class MonolithicSchedulerContext:
    cluster_config: ClusterConfig
    cluster: Cluster
    cluster_scheduler: BaseClusterScheduler
    predictor: BaseExecutionTimePredictor
    request_generator_config: SyntheticRequestGeneratorConfig

    @property
    def cluster_type(self) -> ClusterType:
        return ClusterType.MONOLITHIC

    def replica_scheduler_keys(self) -> list[tuple[int, int]]:
        return list(self.cluster_scheduler._dp_replica_schedulers.keys())

    def get_replica_scheduler(self, replica_id: int, dp_id: int):
        return self.cluster_scheduler.get_dp_replica_scheduler(replica_id, dp_id)


def build_monolithic_context(
    *,
    replica_scheduler_kind: ReplicaSchedulerKind = ReplicaSchedulerKind.VLLM_V1,
    num_replicas: int = 1,
    attn_data_parallel_size: int = 1,
    dummy_execution_time_ms: float = 100.0,
    enable_dummy_mode: bool = True,
    batch_size_cap: int = 128,
    max_tokens_in_batch: int = 4096,
    num_blocks: int = 10_000,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "a100",
    network_device: str | None = None,
    metrics_output_dir: Optional[Path] = None,
) -> MonolithicSchedulerContext:
    if replica_scheduler_kind == ReplicaSchedulerKind.VLLM_V1:
        replica_scheduler_config = VllmV1SchedulerConfig(
            batch_size_cap=batch_size_cap,
            max_tokens_in_batch=max_tokens_in_batch,
            num_blocks=num_blocks,
            num_blocks_mode="explicit",
        )
    elif replica_scheduler_kind == ReplicaSchedulerKind.SGLANG:
        replica_scheduler_config = SglangSchedulerConfig(
            batch_size_cap=batch_size_cap,
            max_tokens_in_batch=max_tokens_in_batch,
            num_blocks=num_blocks,
            num_blocks_mode="explicit",
        )
    else:
        raise ValueError(f"Unsupported replica scheduler kind: {replica_scheduler_kind}")

    net_dev = network_device
    if net_dev is None:
        dev = str(device).lower()
        if dev.startswith("a100"):
            net_dev = "a100_pairwise_nvlink"
        elif dev.startswith("a800"):
            net_dev = "a800_pairwise_nvlink"
        elif dev.startswith("h800") or dev.startswith("h100"):
            # Profiling compute may be under h800; network fixtures commonly use h100.
            net_dev = "h100_pairwise_nvlink"
        else:
            net_dev = "a100_pairwise_nvlink"

    cluster_config = ClusterConfig(
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=replica_scheduler_config,
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=bool(enable_dummy_mode),
            dummy_execution_time_ms=dummy_execution_time_ms,
        ),
        num_replicas=num_replicas,
        replica_config=ReplicaConfig(
            model_name=model_name,
            device=str(device),
            network_device=str(net_dev),
            num_pipeline_stages=1,
            attn_tensor_parallel_size=1,
            attn_data_parallel_size=attn_data_parallel_size,
        ),
    )

    metrics_config = MetricsConfig(
        output_dir=str(metrics_output_dir or Path.cwd() / "hybridsim_scheduler_metrics"),
        write_metrics=False,
        write_json_trace=False,
        enable_chrome_trace=False,
    )
    request_generator_config = SyntheticRequestGeneratorConfig()

    cluster = Cluster(
        cluster_config,
        metrics_config,
        request_generator_config,
    )

    predictor = ExecutionTimePredictorRegistry.get(
        cluster_config.execution_time_predictor_config.get_type(),
        predictor_config=cluster_config.execution_time_predictor_config,
        replica_config=cluster_config.replica_config,
        replica_scheduler_config=cluster_config.replica_scheduler_config,
        metrics_config=metrics_config,
        cluster_config=cluster_config,
        model_manager=None,
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=cluster.cc_backend,
    )

    cluster_scheduler = ClusterSchedulerRegistry.get(
        cluster_config.cluster_scheduler_config.get_type(),
        config=cluster_config,
        cluster=cluster,
        request_generator_config=request_generator_config,
        predictor=predictor,
        available_clusters={ClusterType.MONOLITHIC},
    )

    return MonolithicSchedulerContext(
        cluster_config=cluster_config,
        cluster=cluster,
        cluster_scheduler=cluster_scheduler,
        predictor=predictor,
        request_generator_config=request_generator_config,
    )
