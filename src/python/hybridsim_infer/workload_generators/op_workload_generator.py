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
    """Summarize phase and per-request token / KV lengths from a ScheduleBatch.

    Prefill and decode keep **separate** per-request lists (varlen / no pad):
    - ``prefill_chunk_lens[i]``, ``prefill_cached_lens[i]``
    - ``decode_token_lens[i]``, ``decode_kv_lens[i]``

    Scalar ``cached_prefix_tokens`` / ``cached_decode_tokens`` are sums of the
    prefill-only / decode-only lists respectively (not mixed together).
    """
    prefill_chunk_lens: list[int] = []
    prefill_cached_lens: list[int] = []
    decode_token_lens: list[int] = []
    decode_kv_lens: list[int] = []
    req_ids: set[int] = set()

    def _add_prefill(req: Any, n: int) -> None:
        chunk = max(0, int(n))
        if chunk <= 0:
            return
        hit = int(getattr(req, "num_computed_tokens", 0) or 0) if req is not None else 0
        prefill_chunk_lens.append(chunk)
        prefill_cached_lens.append(max(0, hit))
        if req is not None:
            req_ids.add(int(req.request_id))

    def _add_decode(req: Any, n: int) -> None:
        tokens = max(1, int(n))
        if req is None:
            decode_token_lens.append(tokens)
            decode_kv_lens.append(1)
            return
        computed = int(getattr(req, "num_computed_tokens", 0) or 0)
        prompt = int(getattr(req, "num_prefill_tokens", 0) or 0)
        decode_token_lens.append(tokens)
        decode_kv_lens.append(max(1, max(computed, prompt)))
        req_ids.add(int(req.request_id))

    chunks = list(batch.chunks or [])
    if chunks:
        for ch in chunks:
            if isinstance(ch, PrefillChunk):
                _add_prefill(ch.request, ch.num_tokens)
            elif isinstance(ch, DecodeChunk):
                _add_decode(ch.request, getattr(ch, "num_tokens", 1) or 1)
            else:
                n = int(getattr(ch, "num_tokens", 0) or 0)
                req = getattr(ch, "request", None)
                if req is not None and not bool(
                    getattr(req, "num_computed_tokens", 0)
                    >= getattr(req, "num_prefill_tokens", 0)
                ):
                    _add_prefill(req, n)
                else:
                    _add_decode(req, n or 1)
    else:
        for req in batch.requests or []:
            n = int((batch.tokens_per_request or {}).get(req.request_id, 0))
            computed = int(getattr(req, "num_computed_tokens", 0) or 0)
            prompt = int(getattr(req, "num_prefill_tokens", 0) or 0)
            if computed < prompt:
                _add_prefill(req, n or max(0, prompt - computed))
            else:
                _add_decode(req, n or 1)

    prefill_tokens = sum(prefill_chunk_lens)
    decode_tokens = sum(decode_token_lens)
    cached_prefix_tokens = sum(prefill_cached_lens)
    cached_decode_tokens = sum(decode_kv_lens)

    if prefill_tokens > 0 and decode_tokens > 0:
        phase = BatchPhase.MIXED
    elif decode_tokens > 0:
        phase = BatchPhase.DECODE
    else:
        phase = BatchPhase.PREFILL

    return BatchFeatures(
        phase=phase,
        num_tokens=max(0, prefill_tokens + decode_tokens),
        num_prefill_tokens=max(0, prefill_tokens),
        num_decode_tokens=max(0, decode_tokens),
        batch_size=max(1, len(req_ids) or len(batch.requests or [])),
        cached_decode_tokens=max(0, cached_decode_tokens),
        cached_prefix_tokens=max(0, cached_prefix_tokens),
        prefill_chunk_lens=prefill_chunk_lens,
        prefill_cached_lens=prefill_cached_lens,
        decode_token_lens=decode_token_lens,
        decode_kv_lens=decode_kv_lens,
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
