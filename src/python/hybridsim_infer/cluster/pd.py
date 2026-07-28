"""PD cluster: Prefill pool for arrivals, Decode pool for handoffs."""

from __future__ import annotations

from typing import Sequence

from hybridsim_infer.cluster.base import ClusterManager
from hybridsim_infer.request import InferenceRequest


class PdClusterManager(ClusterManager):
    def __init__(
        self,
        *,
        prefill_replica_ids: Sequence[int],
        decode_replica_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self._prefill_ids = [int(i) for i in prefill_replica_ids]
        self._decode_ids = [int(i) for i in decode_replica_ids]
        if not self._prefill_ids:
            raise ValueError("PdClusterManager requires non-empty prefill pool")
        if not self._decode_ids:
            raise ValueError("PdClusterManager requires non-empty decode pool")

    def bind_replicas(self, replicas) -> None:
        super().bind_replicas(replicas)
        n = len(self._replicas)
        for i in self._prefill_ids + self._decode_ids:
            if i < 0 or i >= n:
                raise RuntimeError(
                    f"PD pool replica id {i} out of range [0, {n})"
                )

    def on_arrive(self, request: InferenceRequest) -> int:
        rid = self._least_loaded(self._prefill_ids)
        params = dict(request.kv_transfer_params or {})
        params.setdefault("transfer_id", self._alloc_transfer_id())
        params["do_remote_decode"] = True
        params["do_remote_prefill"] = False
        # Hint: any decode peer; concrete D chosen at handoff.
        params["remote_replica_id"] = self._decode_ids[0]
        request.kv_transfer_params = params
        self._loads[rid] += 1
        return rid

    def on_handoff(
        self,
        request: InferenceRequest,
        *,
        from_replica_id: int,
        transfer_id: str = "",
    ) -> int:
        from_rid = int(from_replica_id)
        if 0 <= from_rid < len(self._loads):
            self._loads[from_rid] = max(0, self._loads[from_rid] - 1)

        decode_rid = self._least_loaded(self._decode_ids)
        params = dict(request.kv_transfer_params or {})
        if transfer_id:
            params["transfer_id"] = transfer_id
        params.setdefault("transfer_id", self._alloc_transfer_id())
        params["do_remote_prefill"] = True
        params["do_remote_decode"] = False
        params["remote_replica_id"] = from_rid  # source Prefill
        request.kv_transfer_params = params

        # Decode will remote-load prompt KV from scratch.
        request.num_computed_tokens = 0
        request.num_output_tokens = 0
        request.pending_remote_tokens = 0
        request.pending_lookup = False
        request.lookup_result = None
        request.completed = False

        self._loads[decode_rid] += 1
        return decode_rid
