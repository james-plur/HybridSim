"""Cluster-level scheduling actor."""

from __future__ import annotations

from typing import Any, Callable, Optional

from frontier.types import ClusterType

from hybridsim_scheduler.actor_base import ActorBase, on
from hybridsim_scheduler.frontier_bridge.factory import SchedulerBundle
from hybridsim_scheduler.messages import ClusterScheduleMsg, RequestArrivalMsg
from hybridsim_scheduler.schedule_trace import ScheduleTraceRecorder


class ClusterSchedulerActor(ActorBase):
    """Routes requests to replica schedulers and triggers replica-level batching."""

    def __init__(
        self,
        *,
        sim,
        hs_actor,
        bundle: SchedulerBundle,
        cluster_type: ClusterType,
        message_types: dict[str, Any],
        replica_actors: dict[tuple[int, int], Any],
        trace: Optional[ScheduleTraceRecorder] = None,
        on_request_arrival: Optional[Callable[[], None]] = None,
    ) -> None:
        self._bundle = bundle
        self._cluster_scheduler = bundle.cluster_scheduler(cluster_type)
        self._cluster_type = cluster_type
        self._replica_actors = replica_actors
        self._trace = trace
        self._on_request_arrival = on_request_arrival
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def send_request_arrival(self, request) -> None:
        self.send(RequestArrivalMsg, request=request)

    def send_request_arrival_at(self, when: float, request) -> None:
        self.send_at(when, RequestArrivalMsg, request=request)

    def send_cluster_schedule(self) -> None:
        self.send(ClusterScheduleMsg, cluster_type=self._cluster_type)

    @on(RequestArrivalMsg)
    def handle_request_arrival(self, _actor, msg) -> None:
        request = msg.request
        now = self.sim.now()
        request.on_arrival(now, self._cluster_type)
        self._cluster_scheduler.add_request(request)
        if self._trace is not None:
            self._trace.record_instant(
                name="RequestArrival",
                time_s=now,
                cluster_type=self._cluster_type.name,
                request_id=request.id,
                args={
                    "prefill_tokens": request.num_prefill_tokens,
                    "decode_tokens": request.num_decode_tokens,
                },
            )
        if self._on_request_arrival is not None:
            self._on_request_arrival()
        self._trigger_cluster_schedule(now)

    @on(ClusterScheduleMsg)
    def handle_cluster_schedule(self, _actor, _msg) -> None:
        self._trigger_cluster_schedule(self.sim.now())

    def _trigger_cluster_schedule(self, now: float) -> None:
        if self._trace is not None:
            self._trace.record_instant(
                name="ClusterSchedule",
                time_s=now,
                cluster_type=self._cluster_type.name,
            )
        request_mapping = self._cluster_scheduler.schedule()
        affected_pairs: set[tuple[int, int]] = set()

        for replica_id, dp_id, request in request_mapping:
            affected_pairs.add((replica_id, dp_id))
            if request is None:
                continue
            if self._cluster_type in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
                request.bind_thinking_home_queue(
                    self._cluster_type, replica_id, dp_id
                )
            self._cluster_scheduler.get_dp_replica_scheduler(
                replica_id, dp_id
            ).add_request(request)

        for replica_id, dp_id in affected_pairs:
            replica_actor = self._replica_actors.get((replica_id, dp_id))
            if replica_actor is not None:
                replica_actor.send_replica_schedule()
