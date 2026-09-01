"""Assemble inference simulation topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hybridsim import Simulation, create_request_profile_session

from hybridsim_infer.actors.cluster import ClusterActor
from hybridsim_infer.actors.kv_store import KvStoreActor
from hybridsim_infer.actors.replica import ReplicaActor
from hybridsim_infer.config import InferenceConfig
from hybridsim_infer.messages import INFER_MESSAGE_TYPES
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators.base import RequestGenerator
from hybridsim_infer.results import (
    config_to_dict,
    request_record,
    resolve_artifact_path,
    summarize_metrics,
    write_json,
    write_jsonl,
)


@dataclass
class InferenceSimulation:
    """Handle returned by ``build_inference_simulation``."""

    sim: Simulation
    cluster: ClusterActor
    replicas: list[ReplicaActor]
    config: InferenceConfig
    kv_store: Optional[KvStoreActor] = None
    network: Any = field(default=None, repr=False)
    profile: Any = field(default=None, repr=False)

    def schedule_arrivals(self, requests: list[InferenceRequest]) -> None:
        self.cluster.schedule_arrivals(requests)

    def schedule_from_generator(
        self, generator: RequestGenerator
    ) -> list[InferenceRequest]:
        """Generate requests then inject arrivals via ``schedule_arrivals``."""
        requests = generator.generate()
        self.schedule_arrivals(requests)
        return requests

    def metrics(self) -> dict[str, Any]:
        """Aggregate TTFT / TPS / hit-rate from finished requests."""
        return summarize_metrics(
            list(self.finished_requests),
            n_scheduled=int(self.cluster.arrived_count),
            sim_now=float(self.now),
        )

    def write_outputs(self) -> dict[str, Path]:
        """Write enabled artifacts (metrics / requests / config snapshot)."""
        out = self.config.output
        written: dict[str, Path] = {}

        metrics_path = resolve_artifact_path(
            enabled=out.metrics.enabled,
            path=out.metrics.path,
            output_dir=out.dir,
            default_name="metrics.json",
        )
        if metrics_path is not None:
            write_json(metrics_path, self.metrics())
            written["metrics"] = metrics_path

        requests_path = resolve_artifact_path(
            enabled=out.requests.enabled,
            path=out.requests.path,
            output_dir=out.dir,
            default_name="requests.jsonl",
        )
        if requests_path is not None:
            write_jsonl(
                requests_path,
                [request_record(req) for req in self.finished_requests],
            )
            written["requests"] = requests_path

        snapshot_path = resolve_artifact_path(
            enabled=out.config_snapshot.enabled,
            path=out.config_snapshot.path,
            output_dir=out.dir,
            default_name="config.json",
        )
        if snapshot_path is not None:
            write_json(snapshot_path, config_to_dict(self.config))
            written["config_snapshot"] = snapshot_path

        return written

    def run(self) -> None:
        try:
            self.sim.run()
        finally:
            if self.profile is not None:
                self.profile.stop()
            self.write_outputs()

    def check_errors(self) -> None:
        self.sim.check_errors()

    @property
    def finished_requests(self) -> list[InferenceRequest]:
        return self.cluster.finished_requests

    @property
    def now(self) -> float:
        return self.sim.now

    @property
    def profile_path(self):
        if self.profile is None:
            return None
        return getattr(self.profile, "output_path", None)


def build_inference_simulation(
    config: InferenceConfig | None = None,
) -> InferenceSimulation:
    """Build Cluster + N homogeneous Replica(+WorkerEngine); optional shared Store."""
    if config is None:
        config = InferenceConfig()
    config.validate()

    profile_out = config.output.request_profile
    profile = create_request_profile_session(
        enabled=bool(profile_out.enabled),
        request_profile_path=profile_out.path,
        request_profile_dir=profile_out.dir,
    )
    profile_arg = profile if getattr(profile, "enabled", False) else None

    sim = Simulation(config)
    sim.register_messages(list(INFER_MESSAGE_TYPES))

    cluster = sim.spawn_actor(ClusterActor, config=config, profile=profile_arg)

    kv_store: Optional[KvStoreActor] = None
    if config.kv.enable_store:
        kv_store = sim.spawn_actor(KvStoreActor, config=config)

    ns = config.network_sim
    network = None
    ranks = 1
    if ns.enabled:
        op_level = config.resolved_op_level()
        ranks = ns.resolved_ranks(op_level.parallel)
        n_replicas = config.cluster.resolved_num_replicas()
        addrs = [
            (rid, rank)
            for rid in range(n_replicas)
            for rank in range(ranks)
        ]
        network = sim.create_network(
            addrs,
            topology=ns.resolved_topology(),
            routing=ns.resolved_routing(),
            layers=ns.resolved_layers(),
            num_leaf=int(ns.num_leaf),
            num_spine=int(ns.num_spine),
            leaf_downlinks=int(ns.leaf_downlinks),
            leaf_uplinks=int(ns.leaf_uplinks),
            link_bandwidth_bps=float(ns.link_bandwidth_bps),
            link_delay_s=float(ns.link_delay_s),
            bw_policy=ns.resolved_bw_policy(),
            lb_policy=ns.resolved_lb_policy(),
            seed=int(ns.seed),
        )

    replicas: list[ReplicaActor] = []
    for rid in range(config.cluster.resolved_num_replicas()):
        if network is not None:
            engines = [sim.create_engine_actor() for _ in range(ranks)]
            for rank, eng in enumerate(engines):
                eng.install_network(network, rid, rank)
            replica = sim.spawn_actor(
                ReplicaActor,
                config=config,
                replica_id=rid,
                cluster=cluster,
                engines=engines,
                kv_store=kv_store,
                kv_engine=(
                    sim.create_engine_actor() if config.kv.enable_store else None
                ),
                profile=profile_arg,
            )
        else:
            replica = sim.spawn_actor(
                ReplicaActor,
                config=config,
                replica_id=rid,
                cluster=cluster,
                engine=sim.create_engine_actor(),
                kv_store=kv_store,
                kv_engine=(
                    sim.create_engine_actor() if config.kv.enable_store else None
                ),
                profile=profile_arg,
            )
        replicas.append(replica)

    cluster.set_replicas(replicas)
    return InferenceSimulation(
        sim=sim,
        cluster=cluster,
        replicas=replicas,
        config=config,
        kv_store=kv_store,
        network=network,
        profile=profile,
    )
