"""Monolith cluster: least-load over all replicas."""

from __future__ import annotations

from hybridsim_infer.cluster.base import ClusterManager
from hybridsim_infer.request import InferenceRequest


class MonolithClusterManager(ClusterManager):
    def on_arrive(self, request: InferenceRequest) -> int:
        if not self._replicas:
            raise RuntimeError("MonolithClusterManager has no replicas")
        rid = self._least_loaded(range(len(self._replicas)))
        self._loads[rid] += 1
        # Clear PD stamps if any (monolith path).
        params = dict(request.kv_transfer_params or {})
        params.pop("do_remote_decode", None)
        params.pop("do_remote_prefill", None)
        if params:
            request.kv_transfer_params = params
        else:
            request.kv_transfer_params = None
        return rid

    def on_handoff(
        self,
        request: InferenceRequest,
        *,
        from_replica_id: int,
        transfer_id: str = "",
    ) -> int:
        raise RuntimeError(
            "MonolithClusterManager does not support RequestHandoffMsg"
        )
