"""Wrap Frontier ``BaseExecutionTimePredictor`` for hybridsim ``ScheduleBatch``."""

from __future__ import annotations

from typing import Optional

from frontier.entities import Batch, Request
from frontier.execution_time_predictor import BaseExecutionTimePredictor
from frontier.types import ClusterType

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import ScheduleBatch


class FrontierBatchDurationPredictor:
    """``BatchDurationPredictor`` that adapts ``ScheduleBatch`` then calls Frontier RF.

    Adapt logic lives here (not on Replica). Prediction is delegated to an already
    constructed Frontier ``BaseExecutionTimePredictor`` (typically RandomForest,
    non-dummy).
    """

    def __init__(
        self,
        frontier_predictor: BaseExecutionTimePredictor,
        *,
        cluster_type: ClusterType = ClusterType.MONOLITHIC,
        replica_id: int = 0,
        is_moe: bool = False,
    ) -> None:
        self._fp = frontier_predictor
        self._cluster_type = cluster_type
        self._replica_id = int(replica_id)
        self._is_moe = bool(is_moe)
        self.last_adapted_batch: Optional[Batch] = None

    @property
    def frontier_predictor(self) -> BaseExecutionTimePredictor:
        return self._fp

    @property
    def cluster_type(self) -> ClusterType:
        return self._cluster_type

    def _adapt_request(
        self, req: InferenceRequest, scheduled_tokens: int
    ) -> Request:
        _ = scheduled_tokens
        fr = Request(
            arrived_at=float(req.arrived_at),
            num_prefill_tokens=int(req.num_prefill_tokens),
            num_decode_tokens=int(req.num_decode_tokens),
            num_processed_tokens=int(req.num_computed_tokens),
        )
        if int(req.num_computed_tokens) >= int(req.num_prefill_tokens):
            fr._is_prefill_complete = True
        return fr

    def _scheduled_tokens(self, batch: ScheduleBatch, req: InferenceRequest) -> int:
        rid = int(req.request_id)
        if rid in batch.tokens_per_request:
            return int(batch.tokens_per_request[rid])
        for chunk in batch.chunks:
            chunk_req = getattr(chunk, "request", None)
            if chunk_req is not None and int(chunk_req.request_id) == rid:
                return int(getattr(chunk, "num_tokens", 0))
        return 0

    def _adapt(self, batch: ScheduleBatch) -> Batch:
        requests: list[Request] = []
        num_tokens: list[int] = []
        for req in batch.requests:
            n = self._scheduled_tokens(batch, req)
            requests.append(self._adapt_request(req, n))
            num_tokens.append(n)
        if not requests:
            raise ValueError("ScheduleBatch has no requests to adapt")
        return Batch(
            self._replica_id,
            requests,
            num_tokens,
            is_moe=self._is_moe,
        )

    def predict(self, schedule_batch: ScheduleBatch) -> float:
        """Return stage execution time in seconds (Frontier ``ExecutionTime.total_time``)."""
        fb = self._adapt(schedule_batch)
        self.last_adapted_batch = fb
        num_layers = int(self._fp._num_layers_per_pipeline_stage)
        execution_time = self._fp.predict_stage_execution_time(
            fb,
            stage_id=0,
            cluster_type=self._cluster_type,
            num_layers=num_layers,
        )
        return float(execution_time.total_time)
