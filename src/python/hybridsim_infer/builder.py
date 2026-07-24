"""Assemble NO_NETWORK inference simulation topology."""

from __future__ import annotations

from dataclasses import dataclass

from hybridsim import Simulation

from hybridsim_infer.actors.cluster_scheduler import ClusterSchedulerActor
from hybridsim_infer.actors.replica_scheduler import ReplicaSchedulerActor
from hybridsim_infer.config import InferenceConfig
from hybridsim_infer.kv_cache import KvCacheManager
from hybridsim_infer.messages import INFER_MESSAGE_TYPES
from hybridsim_infer.request import InferenceRequest


@dataclass
class InferenceSimulation:
    """Handle returned by ``build_inference_simulation``."""

    sim: Simulation
    cluster: ClusterSchedulerActor
    replicas: list[ReplicaSchedulerActor]
    config: InferenceConfig

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
    """Build Cluster + N Replica(+WorkerEngine) with no Network / KV actors."""
    if config is None:
        config = InferenceConfig()

    sim = Simulation(config)
    sim.register_messages(list(INFER_MESSAGE_TYPES))

    cluster = sim.spawn_actor(ClusterSchedulerActor)

    replicas: list[ReplicaSchedulerActor] = []
    for rid in range(int(config.num_replicas)):
        engine = sim.create_engine_actor()
        kv = KvCacheManager(
            num_gpu_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        replica = sim.spawn_actor(
            ReplicaSchedulerActor,
            replica_id=rid,
            cluster=cluster,
            engine=engine,
            kv_cache_manager=kv,
            step_interval=config.step_interval,
            dummy_exec_s=config.dummy_exec_s,
            tokens_per_step=config.tokens_per_step,
        )
        replicas.append(replica)

    cluster.set_replicas(replicas)
    return InferenceSimulation(
        sim=sim, cluster=cluster, replicas=replicas, config=config
    )
