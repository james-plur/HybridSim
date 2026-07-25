"""ClusterSchedulerActor: arrival injection + finish tracking + dispatch."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.messages import (
    RequestArriveMsg,
    RequestFinishMsg,
    RequestHandoffMsg,
    RequestMsg,
)
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
        kv_mode: str = "store",
        kv_p2p_prefill_replica: int = 0,
        kv_p2p_decode_replica: int = 1,
    ) -> None:
        self._replicas: list[Any] = list(replicas or [])
        self._replica_loads: list[int] = [0] * len(self._replicas)
        self.finished_requests: list[InferenceRequest] = []
        self.arrived_count = 0
        self._kv_mode = str(kv_mode)
        self._p2p_prefill = int(kv_p2p_prefill_replica)
        self._p2p_decode = int(kv_p2p_decode_replica)
        self._next_transfer_id = 1
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def set_replicas(self, replicas: list[Any]) -> None:
        self._replicas = list(replicas)
        self._replica_loads = [0] * len(self._replicas)

    def schedule_arrivals(self, requests: list[InferenceRequest]) -> None:
        """Inject RequestArriveMsg at each request's arrived_at time."""
        for req in requests:
            self.send_at(float(req.arrived_at), RequestArriveMsg, request=req)

    def _alloc_transfer_id(self) -> str:
        tid = f"xfer-{self._next_transfer_id}"
        self._next_transfer_id += 1
        return tid

    @on(RequestArriveMsg)
    def on_request_arrive(self, _actor, msg: RequestArriveMsg) -> None:
        self.arrived_count += 1
        if not self._replicas:
            raise RuntimeError("ClusterScheduler has no replicas")

        req = msg.request
        if self._kv_mode == "p2p":
            rid = self._p2p_prefill
            if rid < 0 or rid >= len(self._replicas):
                raise RuntimeError(f"invalid kv_p2p_prefill_replica={rid}")
            params = dict(req.kv_transfer_params or {})
            params.setdefault("transfer_id", self._alloc_transfer_id())
            params["do_remote_decode"] = True
            params["do_remote_prefill"] = False
            params["remote_replica_id"] = self._p2p_decode
            req.kv_transfer_params = params
        else:
            rid = dispatch(req, self._replica_loads)

        self._replica_loads[rid] += 1
        replica = self._replicas[rid]
        replica.send(RequestMsg, request=req)

    @on(RequestHandoffMsg)
    def on_request_handoff(self, _actor, msg: RequestHandoffMsg) -> None:
        """Prefill finished → route to Decode with do_remote_prefill."""
        req = msg.request
        from_rid = int(msg.from_replica_id)
        if 0 <= from_rid < len(self._replica_loads):
            self._replica_loads[from_rid] = max(0, self._replica_loads[from_rid] - 1)

        decode_rid = self._p2p_decode
        if decode_rid < 0 or decode_rid >= len(self._replicas):
            raise RuntimeError(f"invalid kv_p2p_decode_replica={decode_rid}")

        params = dict(req.kv_transfer_params or {})
        if msg.transfer_id:
            params["transfer_id"] = msg.transfer_id
        params.setdefault("transfer_id", self._alloc_transfer_id())
        params["do_remote_prefill"] = True
        params["do_remote_decode"] = False
        params["remote_replica_id"] = from_rid
        req.kv_transfer_params = params
        # Decode will remote-load prompt KV from scratch.
        req.num_computed_tokens = 0
        req.num_output_tokens = 0
        req.pending_remote_tokens = 0
        req.pending_lookup = False
        req.lookup_result = None
        req.completed = False
        req.status = req.status  # reset below via RequestMsg handler

        self._replica_loads[decode_rid] += 1
        self._replicas[decode_rid].send(RequestMsg, request=req)

    @on(RequestFinishMsg)
    def on_request_finish(self, _actor, msg: RequestFinishMsg) -> None:
        self.finished_requests.append(msg.request)
        rid = int(msg.replica_id)
        if 0 <= rid < len(self._replica_loads):
            self._replica_loads[rid] = max(0, self._replica_loads[rid] - 1)
