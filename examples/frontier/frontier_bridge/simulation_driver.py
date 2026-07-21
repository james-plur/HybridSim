"""Build a platform Simulation with Frontier actors/messages registered."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from frontier.entities import Request
from frontier.types import ClusterType

from hybridsim import ScheduleTraceRecorder, Simulation as PlatformSimulation

from frontier_bridge.batch_completion import handle_kv_transfer_complete
from frontier_bridge.cluster_scheduler_actor import ClusterSchedulerActor
from frontier_bridge.config import (
    ArchitectureConfig,
    MonolithicConfig,
    load_frontier_config_from_cli_args,
)
from frontier_bridge.factory import build_scheduler_bundle
from frontier_bridge.messages import (
    BatchCompleteMsg,
    ClusterScheduleMsg,
    KVTransferCompleteMsg,
    ReplicaScheduleMsg,
    RequestArrivalMsg,
)
from frontier_bridge.replica_scheduler_actor import ReplicaSchedulerActor

FrontierConfig = Union[ArchitectureConfig, MonolithicConfig]


class FrontierSimulation(PlatformSimulation):
    """Platform Simulation with Frontier scheduler actors and request helpers."""

    def __init__(self, config: FrontierConfig) -> None:
        super().__init__(config)
        self.frontier_config = config
        self.frontier = config.to_frontier()
        self.bundle = build_scheduler_bundle(self.frontier)
        self.register_messages(
            [
                RequestArrivalMsg,
                ClusterScheduleMsg,
                ReplicaScheduleMsg,
                BatchCompleteMsg,
                KVTransferCompleteMsg,
            ]
        )
        self.trace = ScheduleTraceRecorder(
            source="hybridsim",
            run_dir=config.trace_output_dir,
            metadata={
                "sys_arch": self.frontier.sys_arch,
                "simulation_mode": self.frontier.simulation_mode,
            },
        )

        self._requests: list[Request] = []
        self._clusters: dict[ClusterType, ClusterSchedulerActor] = {}
        self._replicas: dict[tuple[ClusterType, int, int], ReplicaSchedulerActor] = {}

        for cluster_type in self.bundle.clusters:
            self._setup_cluster(cluster_type)

        self.before_run = self._schedule_arrivals

    def _setup_cluster(self, cluster_type: ClusterType) -> None:
        replicas_for_cluster: dict[tuple[int, int], ReplicaSchedulerActor] = {}

        for replica_id, dp_id in self.bundle.replica_scheduler_keys(cluster_type):
            engine = self.create_engine_actor()
            hs_replica_actor = self.create_hs_actor()
            replica_scheduler = self.bundle.get_replica_scheduler(
                cluster_type, replica_id, dp_id
            )
            replica = ReplicaSchedulerActor(
                sim=self.hs_sim,
                hs_actor=hs_replica_actor,
                engine=engine,
                replica_scheduler=replica_scheduler,
                predictor=self.bundle.predictor(cluster_type),
                cluster_type=cluster_type,
                cluster_scheduler=self.bundle.cluster_scheduler(cluster_type),
                replica_id=replica_id,
                dp_id=dp_id,
                message_types=self.message_types,
                trace=self.trace,
                kv_cache_transfer_predictor=self.bundle.kv_cache_transfer_predictor,
                on_kv_transfer=self._on_kv_transfer_complete,
            )
            self.add_actor(replica)
            replicas_for_cluster[(replica_id, dp_id)] = replica
            self._replicas[(cluster_type, replica_id, dp_id)] = replica

        cluster = ClusterSchedulerActor(
            sim=self.hs_sim,
            hs_actor=self.create_hs_actor(),
            bundle=self.bundle,
            cluster_type=cluster_type,
            message_types=self.message_types,
            replica_actors=replicas_for_cluster,
            trace=self.trace,
        )
        self.add_actor(cluster)
        self._clusters[cluster_type] = cluster

    def _on_kv_transfer_complete(self, transfer_info) -> None:
        clusters = handle_kv_transfer_complete(
            time_s=self.hs_sim.now(),
            transfer_info=transfer_info,
            global_scheduler=self.bundle.global_scheduler,
        )
        for cluster_type in clusters:
            self._clusters[cluster_type].send_cluster_schedule()

    def _arrival_cluster(self) -> ClusterType:
        if self.bundle.is_disaggregated:
            return ClusterType.PREFILL
        return ClusterType.MONOLITHIC

    def generate_requests(self) -> list[Request]:
        requests = self.bundle.request_generator.generate()
        self._requests = list(requests)
        return self._requests

    def add_request(
        self,
        *,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
    ) -> Request:
        request = Request(
            arrived_at=arrived_at,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
        )
        self._requests.append(request)
        return request

    def inject_request(self, request: Request) -> None:
        if request not in self._requests:
            self._requests.append(request)
        cluster_type = self._arrival_cluster()
        request.set_arrived_at(self.hs_sim.now())
        self._clusters[cluster_type].send_request_arrival(request)
        setattr(request, "_hybridsim_arrival_scheduled", True)

    def inject_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.inject_request(request)

    def _schedule_arrivals(self) -> None:
        if not self._requests:
            requests = sorted(
                self.generate_requests(), key=lambda request: request.arrived_at
            )
        else:
            requests = list(self._requests)

        cluster = self._clusters[self._arrival_cluster()]
        for request in requests:
            if getattr(request, "_hybridsim_arrival_scheduled", False):
                continue
            cluster.send_request_arrival_at(request.arrived_at, request)
            setattr(request, "_hybridsim_arrival_scheduled", True)

    @property
    def requests(self) -> list[Request]:
        return list(self._requests)

    @property
    def predicted_duration_total(self) -> float:
        return sum(
            replica.predicted_duration_total for replica in self._replicas.values()
        )

    @property
    def completed_batches(self) -> int:
        return sum(replica.completed_batches for replica in self._replicas.values())

    def all_requests_completed(self) -> bool:
        return bool(self._requests) and all(
            request.completed for request in self._requests
        )

    def write_trace(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = output_dir / "inference_profile.json"
        self.trace.run_dir = output_dir
        self.trace.write(profile_path)
        return profile_path


def build_frontier_simulation(config: FrontierConfig) -> FrontierSimulation:
    """Register Frontier msgs/actors on a platform Simulation and return it."""
    return FrontierSimulation(config)


def load_config_from_cli_args(cli_args: list[str]):
    return load_frontier_config_from_cli_args(cli_args)


def run_from_cli_args(
    cli_args: list[str],
    *,
    build_dir: Optional[Path] = None,
    trace_output_dir: Optional[Path] = None,
) -> FrontierSimulation:
    config = ArchitectureConfig(
        frontier=load_frontier_config_from_cli_args(cli_args),
        build_dir=build_dir,
        trace_output_dir=trace_output_dir,
    )
    simulation = build_frontier_simulation(config)
    simulation.run()
    simulation.check_errors()
    if trace_output_dir is not None:
        simulation.write_trace(trace_output_dir)
    return simulation
