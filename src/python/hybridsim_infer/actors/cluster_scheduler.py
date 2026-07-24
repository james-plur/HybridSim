"""ClusterSchedulerActor: arrival injection + finish tracking + dispatch."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.messages import RequestArriveMsg, RequestFinishMsg, RequestMsg
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.stubs import dispatch


class ClusterSchedulerActor(ActorBase):
    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        replicas: Optional[list[Any]] = None,
    ) -> None:
        self._replicas: list[Any] = list(replicas or [])
        self._replica_loads: list[int] = [0] * len(self._replicas)
        self.finished_requests: list[InferenceRequest] = []
        self.arrived_count = 0
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def set_replicas(self, replicas: list[Any]) -> None:
        self._replicas = list(replicas)
        self._replica_loads = [0] * len(self._replicas)

    def schedule_arrivals(self, requests: list[InferenceRequest]) -> None:
        """Inject RequestArriveMsg at each request's arrived_at time."""
        for req in requests:
            self.send_at(float(req.arrived_at), RequestArriveMsg, request=req)

    @on(RequestArriveMsg)
    def on_request_arrive(self, _actor, msg: RequestArriveMsg) -> None:
        self.arrived_count += 1
        if not self._replicas:
            raise RuntimeError("ClusterScheduler has no replicas")
        rid = dispatch(msg.request, self._replica_loads)
        self._replica_loads[rid] += 1
        replica = self._replicas[rid]
        replica.send(RequestMsg, request=msg.request)

    @on(RequestFinishMsg)
    def on_request_finish(self, _actor, msg: RequestFinishMsg) -> None:
        self.finished_requests.append(msg.request)
        rid = int(msg.replica_id)
        if 0 <= rid < len(self._replica_loads):
            self._replica_loads[rid] = max(0, self._replica_loads[rid] - 1)
