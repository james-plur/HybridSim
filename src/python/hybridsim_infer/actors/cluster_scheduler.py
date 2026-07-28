"""ClusterSchedulerActor: arrival / handoff / finish; dispatch via ClusterManager."""

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


class ClusterSchedulerActor(ActorBase):
    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        replicas: Optional[list[Any]] = None,
        manager: Optional[ClusterManager] = None,
    ) -> None:
        self._mgr: ClusterManager = manager or MonolithClusterManager()
        self.finished_requests: list[InferenceRequest] = []
        self.arrived_count = 0
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
        rid = self._mgr.on_arrive(req)
        self._mgr.replica(rid).send(RequestMsg, request=req)

    @on(RequestHandoffMsg)
    def on_request_handoff(self, _actor, msg: RequestHandoffMsg) -> None:
        req = msg.request
        decode_rid = self._mgr.on_handoff(
            req,
            from_replica_id=int(msg.from_replica_id),
            transfer_id=str(msg.transfer_id or ""),
        )
        self._mgr.replica(decode_rid).send(RequestMsg, request=req)

    @on(RequestFinishMsg)
    def on_request_finish(self, _actor, msg: RequestFinishMsg) -> None:
        self.finished_requests.append(msg.request)
        self._mgr.on_finish(int(msg.replica_id))
