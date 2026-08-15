"""ClusterActor: arrival / handoff / finish; dispatch via ClusterManager."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.cluster import ClusterManager, MonolithClusterManager
from hybridsim_infer.messages import (
    RequestArriveMsg,
    RequestFinishMsg,
    RequestHandoffMsg,
    RequestMsg,
)
from hybridsim_infer.request import InferenceRequest


class ClusterActor(ActorBase):
    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        replicas: Optional[list[Any]] = None,
        manager: Optional[ClusterManager] = None,
        profile: Any = None,
    ) -> None:
        self._mgr: ClusterManager = manager or MonolithClusterManager()
        self.finished_requests: list[InferenceRequest] = []
        self.arrived_count = 0
        self._profile = profile
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)
        if replicas:
            self.set_replicas(replicas)

    def set_replicas(self, replicas: list[Any]) -> None:
        self._mgr.bind_replicas(replicas)

    @property
    def manager(self) -> ClusterManager:
        return self._mgr

    def schedule_arrivals(self, requests: list[InferenceRequest]) -> None:
        for req in requests:
            self.send_at(float(req.arrived_at), RequestArriveMsg, request=req)

    @on(RequestArriveMsg)
    def on_request_arrive(self, _actor, msg: RequestArriveMsg) -> None:
        self.arrived_count += 1
        req = msg.request
        now = float(self.sim.now())
        if self._profile is not None:
            self._profile.emit_cluster_schedule(time_s=now)
        rid = self._mgr.on_arrive(req)
        if self._profile is not None:
            self._profile.emit_dispatch(
                time_s=now,
                request_id=int(req.request_id),
                replica_id=int(rid),
                kind="arrive",
                request=req,
            )
        self._mgr.replica(rid).send(RequestMsg, request=req)

    @on(RequestHandoffMsg)
    def on_request_handoff(self, _actor, msg: RequestHandoffMsg) -> None:
        req = msg.request
        now = float(self.sim.now())
        if self._profile is not None:
            self._profile.emit_cluster_schedule(time_s=now)
        decode_rid = self._mgr.on_handoff(
            req,
            from_replica_id=int(msg.from_replica_id),
            transfer_id=str(msg.transfer_id or ""),
        )
        if self._profile is not None:
            self._profile.emit_dispatch(
                time_s=now,
                request_id=int(req.request_id),
                replica_id=int(decode_rid),
                kind="handoff",
                request=req,
            )
        self._mgr.replica(decode_rid).send(RequestMsg, request=req)

    @on(RequestFinishMsg)
    def on_request_finish(self, _actor, msg: RequestFinishMsg) -> None:
        self.finished_requests.append(msg.request)
        if self._profile is not None:
            self._profile.emit_request_meta(
                request=msg.request,
                extra={
                    "finished_at": float(self.sim.now()),
                    "finish_replica_id": int(msg.replica_id),
                },
            )
        self._mgr.on_finish(int(msg.replica_id))
