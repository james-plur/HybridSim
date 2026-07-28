"""Assemble NO_NETWORK inference simulation topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hybridsim import Simulation

from hybridsim_infer.actors.cluster_scheduler import ClusterSchedulerActor
from hybridsim_infer.actors.kv_store import KvStoreActor
from hybridsim_infer.actors.replica_scheduler import ReplicaSchedulerActor
from hybridsim_infer.cluster import MonolithClusterManager, PdClusterManager
from hybridsim_infer.config import InferenceConfig
from hybridsim_infer.frameworks import FrameworkFactory
from hybridsim_infer.kv_system import VllmKvCacheManager
from hybridsim_infer.messages import INFER_MESSAGE_TYPES
from hybridsim_infer.request import InferenceRequest


@dataclass
class InferenceSimulation:
    """Handle returned by ``build_inference_simulation``."""

    sim: Simulation
    cluster: ClusterSchedulerActor
    replicas: list[ReplicaSchedulerActor]
    config: InferenceConfig
    kv_store: Optional[KvStoreActor] = None

    def schedule_arrivals(self, requests: list[InferenceRequest]) -> None:
        self.cluster.schedule_arrivals(requests)

    def run(self) -> None:
        self.sim.run()

    def check_errors(self) -> None:
        self.sim.check_errors()

    @property
    def finished_requests(self) -> list[InferenceRequest]:
        return self.cluster.finished_requests

    @property
    def now(self) -> float:
        return self.sim.now


def build_inference_simulation(
    config: InferenceConfig | None = None,
) -> InferenceSimulation:
    """Build Cluster + N homogeneous Replica(+WorkerEngine); optional shared Store."""
    if config is None:
        config = InferenceConfig()

    cluster_type = config.resolved_cluster_type()
    num_replicas = config.resolved_num_replicas()
    if cluster_type == "pd" and num_replicas < 2:
        raise ValueError("cluster_type=pd requires at least 1 Prefill + 1 Decode replica")

    if cluster_type == "pd":
        prefill_ids, decode_ids = config.pd_pools()
        manager = PdClusterManager(
            prefill_replica_ids=prefill_ids,
            decode_replica_ids=decode_ids,
        )
    else:
        manager = MonolithClusterManager()

    sim = Simulation(config)
    sim.register_messages(list(INFER_MESSAGE_TYPES))

    cluster = sim.spawn_actor(ClusterSchedulerActor, manager=manager)

    kv_store: Optional[KvStoreActor] = None
    if config.enable_kv_client:
        kv_store = sim.spawn_actor(
            KvStoreActor,
            num_blocks=config.kv_store_blocks,
            block_size=config.block_size,
        )

    replicas: list[ReplicaSchedulerActor] = []
    for rid in range(num_replicas):
        engine = sim.create_engine_actor()
        kv = VllmKvCacheManager(
            num_gpu_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        kv_engine = (
            sim.create_engine_actor() if config.enable_kv_client else None
        )
        framework = FrameworkFactory.create(
            config.framework,
            tokens_per_step=config.tokens_per_step,
            decode_tokens_per_step=config.decode_tokens_per_step,
            long_prefill_token_threshold=config.long_prefill_token_threshold,
            reserve_full_isl=config.reserve_full_isl,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        replica = sim.spawn_actor(
            ReplicaSchedulerActor,
            replica_id=rid,
            cluster=cluster,
            engine=engine,
            kv_cache_manager=kv,
            kv_store=kv_store,
            kv_engine=kv_engine,
            framework=framework,
            step_interval=config.step_interval,
            dummy_exec_s=config.dummy_exec_s,
            kv_transfer_s=config.kv_transfer_s,
            kv_bandwidth_gbps=config.kv_bandwidth_gbps,
            kv_bytes_per_token=config.kv_bytes_per_token,
            kv_lookup_async=config.kv_lookup_async,
            kv_lookup_rtt_s=config.kv_lookup_rtt_s,
            max_num_scheduled_tokens=config.max_num_scheduled_tokens,
            max_num_running_reqs=config.max_num_running_reqs,
            max_inflight_batches=config.max_inflight_batches,
            duration_mode=config.duration_mode,
            prefill_s_per_token=config.prefill_s_per_token,
            decode_s_per_token=config.decode_s_per_token,
            duration_base_s=config.duration_base_s,
        )
        replicas.append(replica)

    cluster.set_replicas(replicas)
    return InferenceSimulation(
        sim=sim,
        cluster=cluster,
        replicas=replicas,
        config=config,
        kv_store=kv_store,
    )
