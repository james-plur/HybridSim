"""Nested inference simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybridsim.config import SimulationConfig

from hybridsim_infer.config.cluster import ClusterConfig
from hybridsim_infer.config.infer_workload import (
    BatchFixedConfig,
    BatchFrontierConfig,
    BatchLevelConfig,
    BatchTokenProportionalConfig,
    DeviceConfig,
    InferWorkloadConfig,
    ModelConfig,
    NetworkConfig,
    OpLevelConfig,
    ParallelConfig,
)
from hybridsim_infer.config.kv import KvConfig, KvLookupConfig, KvStoreConfig
from hybridsim_infer.config.kv_workload import KvWorkloadConfig
from hybridsim_infer.config.model import ModelSpec
from hybridsim_infer.config.network_sim import NetworkSimConfig
from hybridsim_infer.config.output import ArtifactOutput, OutputConfig, RequestProfileOutput
from hybridsim_infer.config.schedule import (
    ClusterScheduleConfig,
    EngineConfig,
    ReplicaScheduleConfig,
    ScheduleConfig,
)


@dataclass
class InferenceConfig(SimulationConfig):
    """Inference config: nested topology / schedule / KV / workload / optional fabric / output."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    kv: KvConfig = field(default_factory=KvConfig)
    model: ModelSpec = field(default_factory=ModelSpec)
    infer_workload: InferWorkloadConfig = field(default_factory=InferWorkloadConfig)
    kv_workload: KvWorkloadConfig = field(default_factory=KvWorkloadConfig)
    network_sim: NetworkSimConfig = field(default_factory=NetworkSimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` if the nested groups are inconsistent."""
        cluster_type = self.cluster.resolved_cluster_type()
        if cluster_type == "pd":
            np_ = int(self.cluster.num_prefill_replicas)
            nd_ = int(self.cluster.num_decode_replicas)
            if np_ < 1 or nd_ < 1:
                raise ValueError(
                    "cluster.type=pd requires at least 1 Prefill and 1 Decode replica"
                )
        elif int(self.cluster.num_replicas) < 1:
            raise ValueError("cluster.num_replicas must be >= 1")

        policy = (self.schedule.cluster.policy or "").lower().strip()
        if policy != "least_load":
            raise ValueError(
                "schedule.cluster.policy must be 'least_load' "
                f"(only policy implemented), got {self.schedule.cluster.policy!r}"
            )

        self.kv.resolved_store_block_size()
        self.infer_workload.resolved_mode()
        self.infer_workload.resolved_predictor()
        if self.network_sim.enabled:
            self.network_sim.resolved_topology()
            self.network_sim.resolved_routing()
            self.network_sim.resolved_layers()
            self.network_sim.resolved_bw_policy()
            self.network_sim.resolved_lb_policy()

    def resolved_cluster_type(self) -> str:
        return self.cluster.resolved_cluster_type()

    def resolved_num_replicas(self) -> int:
        return self.cluster.resolved_num_replicas()

    def pd_pools(self) -> tuple[list[int], list[int]]:
        return self.cluster.pd_pools()

    def resolved_store_block_size(self) -> int:
        return self.kv.resolved_store_block_size()

    def resolved_op_level(self) -> OpLevelConfig:
        """``infer_workload.op`` with ``model.resolve()`` injected when set."""
        return self.infer_workload.apply_model(self.model.resolve())


__all__ = [
    "ArtifactOutput",
    "BatchFixedConfig",
    "BatchFrontierConfig",
    "BatchLevelConfig",
    "BatchTokenProportionalConfig",
    "ClusterConfig",
    "ClusterScheduleConfig",
    "DeviceConfig",
    "EngineConfig",
    "InferWorkloadConfig",
    "InferenceConfig",
    "ModelConfig",
    "NetworkConfig",
    "OpLevelConfig",
    "ParallelConfig",
    "KvConfig",
    "KvLookupConfig",
    "KvStoreConfig",
    "KvWorkloadConfig",
    "ModelSpec",
    "NetworkSimConfig",
    "OutputConfig",
    "ReplicaScheduleConfig",
    "RequestProfileOutput",
    "ScheduleConfig",
]
