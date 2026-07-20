"""Assemble hybridsim scheduler actors and run Frontier-compatible simulations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

from frontier.config import SimulationConfig as FrontierSimulationConfig
from frontier.entities import Request
from frontier.types import ClusterType

from hybridsim_scheduler.batch_completion import handle_kv_transfer_complete
from hybridsim_scheduler.cluster_scheduler_actor import ClusterSchedulerActor
from hybridsim_scheduler.config import (
    ArchitectureConfig,
    SimulationConfig,
    load_frontier_config_from_cli_args,
)
from hybridsim_scheduler.frontier_bridge.factory import build_scheduler_bundle
from hybridsim_scheduler.messages import register_scheduler_messages
from hybridsim_scheduler.replica_scheduler_actor import ReplicaSchedulerActor
from hybridsim_scheduler.schedule_trace import ScheduleTraceRecorder


def _ensure_hybridsim_py_on_path(build_dir: Optional[Path] = None) -> None:
    if build_dir is None:
        build_dir = Path(__file__).resolve().parents[3] / "build"
    build_dir = build_dir.resolve()
    build_pkg = str(build_dir)
    if build_pkg not in sys.path:
        sys.path.insert(0, build_pkg)


class Simulation:
    """Unified hybridsim entry: ``Simulation(ArchitectureConfig | MonolithicConfig | ...)``."""

    def __init__(self, config: SimulationConfig) -> None:
        _ensure_hybridsim_py_on_path(config.build_dir)
        import hybridsim_py as hs

        self._hs = hs
        self.sim_config = config
        self.config = config.to_frontier()
        self.bundle = build_scheduler_bundle(self.config)
        self.sim = hs.Simulation()
        self.message_types = register_scheduler_messages(self.sim)
        self.trace = ScheduleTraceRecorder(
            source="hybridsim",
            run_dir=config.trace_output_dir,
            metadata={
                "sys_arch": self.config.sys_arch,
                "simulation_mode": self.config.simulation_mode,
            },
        )

        self._requests: list[Request] = []
        self._clusters: dict[ClusterType, ClusterSchedulerActor] = {}
        self._replicas: dict[tuple[ClusterType, int, int], ReplicaSchedulerActor] = {}

        for cluster_type in self.bundle.clusters:
            self._setup_cluster(cluster_type)

    def _setup_cluster(self, cluster_type: ClusterType) -> None:
        hs_actor = self._hs.Actor(self.sim)
        replicas_for_cluster: dict[tuple[int, int], ReplicaSchedulerActor] = {}

        for replica_id, dp_id in self.bundle.replica_scheduler_keys(cluster_type):
            engine = self._hs.EngineActor(self.sim)
            hs_replica_actor = self._hs.Actor(self.sim)
            replica_scheduler = self.bundle.get_replica_scheduler(
                cluster_type, replica_id, dp_id
            )
            replica = ReplicaSchedulerActor(
                sim=self.sim,
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
            replicas_for_cluster[(replica_id, dp_id)] = replica
            self._replicas[(cluster_type, replica_id, dp_id)] = replica

        cluster = ClusterSchedulerActor(
            sim=self.sim,
            hs_actor=hs_actor,
            bundle=self.bundle,
            cluster_type=cluster_type,
            message_types=self.message_types,
            replica_actors=replicas_for_cluster,
            trace=self.trace,
        )
        self._clusters[cluster_type] = cluster

    def _on_kv_transfer_complete(self, transfer_info) -> None:
        clusters = handle_kv_transfer_complete(
            time_s=self.sim.now(),
            transfer_info=transfer_info,
            global_scheduler=self.bundle.global_scheduler,
        )
        for cluster_type in clusters:
            self._clusters[cluster_type].send_cluster_schedule()

    def _arrival_cluster(self) -> ClusterType:
        if self.bundle.is_disaggregated:
            return ClusterType.PREFILL
        return ClusterType.MONOLITHIC

    def _start_actors(self) -> None:
        for cluster in self._clusters.values():
            cluster.start()
        for replica in self._replicas.values():
            replica.start()

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
        """Create a request and append it for later injection / scheduling."""
        request = Request(
            arrived_at=arrived_at,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
        )
        self._requests.append(request)
        return request

    def inject_request(self, request: Request) -> None:
        """Enqueue a RequestArrival at the current simulation time."""
        if request not in self._requests:
            self._requests.append(request)
        cluster_type = self._arrival_cluster()
        request.set_arrived_at(self.sim.now())
        self._clusters[cluster_type].send_request_arrival(request)
        setattr(request, "_hybridsim_arrival_scheduled", True)

    def inject_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.inject_request(request)

    def run(self) -> None:
        """Start actors, schedule arrivals, then drain the DES.

        If ``_requests`` is already populated (e.g. via ``add_request`` /
        ``inject_request``), those requests are used; otherwise requests are
        generated from the Frontier request generator.
        """
        self._start_actors()

        if not self._requests:
            requests = sorted(
                self.generate_requests(), key=lambda request: request.arrived_at
            )
        else:
            requests = list(self._requests)

        arrival_cluster = self._arrival_cluster()
        cluster = self._clusters[arrival_cluster]
        for request in requests:
            if getattr(request, "_hybridsim_arrival_scheduled", False):
                continue
            cluster.send_request_arrival_at(request.arrived_at, request)
            setattr(request, "_hybridsim_arrival_scheduled", True)
        self.sim.run()

    def check_errors(self) -> None:
        for replica in self._replicas.values():
            replica.check_error()
        for cluster in self._clusters.values():
            cluster.check_error()

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


def load_config_from_cli_args(cli_args: list[str]) -> FrontierSimulationConfig:
    """Parse Frontier config from CLI args."""
    return load_frontier_config_from_cli_args(cli_args)


def run_from_cli_args(
    cli_args: list[str],
    *,
    build_dir: Optional[Path] = None,
    trace_output_dir: Optional[Path] = None,
) -> Simulation:
    config = ArchitectureConfig(
        frontier=load_frontier_config_from_cli_args(cli_args),
        build_dir=build_dir,
        trace_output_dir=trace_output_dir,
    )
    simulation = Simulation(config)
    simulation.run()
    simulation.check_errors()
    if trace_output_dir is not None:
        simulation.write_trace(trace_output_dir)
    return simulation
