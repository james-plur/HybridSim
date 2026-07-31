"""ScheduleBatch → Operator DAG → TimeoutKernel workload (analytical)."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch
from hybridsim_infer.workload_generators.analytic_model.configs import (
    AnalyticalConfig,
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.analytic_model.dag_builder import (
    build_operator_dag,
)
from hybridsim_infer.workload_generators.analytic_model.op_analyzer import (
    OpAnalyzer,
    critical_path_duration_s,
)
from hybridsim_infer.workload_generators.analytic_model.types import (
    BatchFeatures,
    BatchPhase,
)
from hybridsim_infer.workload_generators.base import WorkloadGenerator


def extract_batch_features(batch: ScheduleBatch) -> BatchFeatures:
    """Summarize phase and token counts from a ScheduleBatch."""
    prefill_tokens = 0
    decode_tokens = 0
    kv_cache_tokens = 0
    req_ids: set[int] = set()

    chunks = list(batch.chunks or [])
    if chunks:
        for ch in chunks:
            if isinstance(ch, PrefillChunk):
                prefill_tokens += int(ch.num_tokens)
                req_ids.add(int(ch.request.request_id))
                kv_cache_tokens += int(getattr(ch.request, "num_computed_tokens", 0) or 0)
            elif isinstance(ch, DecodeChunk):
                decode_tokens += int(getattr(ch, "num_tokens", 1) or 1)
                req_ids.add(int(ch.request.request_id))
                computed = int(getattr(ch.request, "num_computed_tokens", 0) or 0)
                prompt = int(getattr(ch.request, "num_prefill_tokens", 0) or 0)
                kv_cache_tokens += max(computed, prompt)
            else:
                n = int(getattr(ch, "num_tokens", 0) or 0)
                req = getattr(ch, "request", None)
                if req is not None:
                    req_ids.add(int(req.request_id))
                if req is not None and not bool(
                    getattr(req, "num_computed_tokens", 0)
                    >= getattr(req, "num_prefill_tokens", 0)
                ):
                    prefill_tokens += n
                else:
                    decode_tokens += max(1, n)
    else:
        for req in batch.requests or []:
            req_ids.add(int(req.request_id))
            n = int((batch.tokens_per_request or {}).get(req.request_id, 0))
            computed = int(getattr(req, "num_computed_tokens", 0) or 0)
            prompt = int(getattr(req, "num_prefill_tokens", 0) or 0)
            if computed < prompt:
                prefill_tokens += n or max(0, prompt - computed)
            else:
                decode_tokens += n or 1
                kv_cache_tokens += max(computed, prompt)

    if prefill_tokens > 0 and decode_tokens > 0:
        phase = BatchPhase.MIXED
    elif decode_tokens > 0:
        phase = BatchPhase.DECODE
    else:
        phase = BatchPhase.PREFILL

    num_tokens = prefill_tokens + decode_tokens
    return BatchFeatures(
        phase=phase,
        num_tokens=max(0, num_tokens),
        num_prefill_tokens=max(0, prefill_tokens),
        num_decode_tokens=max(0, decode_tokens),
        batch_size=max(1, len(req_ids) or len(batch.requests or [])),
        kv_cache_tokens=max(0, kv_cache_tokens),
    )


class OpWorkloadGenerator(WorkloadGenerator):
    """Build Operator DAG then analyze into TimeoutKernel Engine workload."""

    def __init__(
        self,
        *,
        analytical: Optional[AnalyticalConfig] = None,
        model: Optional[ModelConfig] = None,
        parallel: Optional[ParallelConfig] = None,
        device: Optional[DeviceConfig] = None,
        network: Optional[NetworkConfig] = None,
        analyzer: Optional[OpAnalyzer] = None,
        expand_attn_sub_kernels: bool = False,
        expand_ffn_sub_kernels: bool = True,
    ) -> None:
        if analytical is not None:
            self._cfg = analytical
        else:
            self._cfg = AnalyticalConfig(
                model=model or ModelConfig(),
                parallel=parallel or ParallelConfig(),
                device=device or DeviceConfig(),
                network=network or NetworkConfig(),
            )
        self._analyzer = analyzer or OpAnalyzer(
            device=self._cfg.device,
            network=self._cfg.network,
            duration_scale=self._cfg.duration_scale,
        )
        self._expand_attn = bool(expand_attn_sub_kernels)
        self._expand_ffn = bool(expand_ffn_sub_kernels)

    @property
    def config(self) -> AnalyticalConfig:
        return self._cfg

    @property
    def analyzer(self) -> OpAnalyzer:
        return self._analyzer

    def build_dag(self, batch: ScheduleBatch):
        features = extract_batch_features(batch)
        return build_operator_dag(
            model=self._cfg.model,
            parallel=self._cfg.parallel,
            batch=features,
            expand_attn_sub_kernels=self._expand_attn,
            expand_ffn_sub_kernels=self._expand_ffn,
        )

    def predict_duration_s(
        self,
        batch: ScheduleBatch,
        *,
        metric: str = "critical_path",
    ) -> float:
        """Analytical duration for ``batch`` (uses config ``duration_scale``)."""
        from hybridsim_infer.workload_generators.analytic_model.op_analyzer import (
            total_kernel_duration_s,
        )

        wl = self(batch, workload_id=0)
        if metric == "sum":
            return total_kernel_duration_s(wl["kernels"])
        return critical_path_duration_s(wl["kernels"])

    def __call__(
        self,
        batch: ScheduleBatch,
        *,
        workload_id: int,
    ) -> dict[str, Any]:
        op_dag = self.build_dag(batch)
        return self._analyzer.analyze(op_dag, workload_id=workload_id)
