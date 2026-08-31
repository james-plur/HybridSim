"""Prefill attention cost respects cached prefix / KV context."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import PrefillChunk, ScheduleBatch
from hybridsim_infer.workload_generators import extract_batch_features
from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
    BatchPhase,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.lower import (
    lower_op,
)
from hybridsim_infer.workload_generators.configs import ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    build_operator_dag,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.fused import (
    FusedAttnOp,
    FusedMlaAttnOp,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    GemmOp,
)
from hybridsim_infer.workload_generators.types import AttnVariant


def _prefill_hit_batch(*, prompt: int, cached: int, chunk: int) -> ScheduleBatch:
    req = InferenceRequest(
        request_id=1,
        arrived_at=0.0,
        num_prefill_tokens=prompt,
        num_decode_tokens=8,
        num_computed_tokens=cached,
    )
    return ScheduleBatch(
        batch_id=1,
        requests=[req],
        tokens_per_request={1: chunk},
        chunks=[PrefillChunk(request=req, num_tokens=chunk)],
    )


def _dense_heads(model: ModelConfig) -> tuple[int, int, int]:
    n_q = max(1, int(model.num_q_heads))
    n_kv = max(1, int(model.num_kv_heads))
    if model.resolved_attn_variant() is AttnVariant.MHA:
        n_kv = n_q
    d = max(1, int(model.get_head_dim()))
    return n_q, n_kv, d


def _attn_prefill_core_flops(
    *,
    chunk: int,
    cached: int,
    model: ModelConfig | None = None,
) -> float:
    model = model or ModelConfig(
        num_layers=1,
        hidden_size=1024,
        intermediate_size=2816,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=64,
        attn_variant=AttnVariant.MHA,
    )
    ctx = max(chunk, cached + chunk)
    if model.resolved_attn_variant().value in ("mla", "dsa"):
        n_q = max(1, int(model.num_q_heads))
        latent = max(1, int(model.kv_lora_rank)) + max(1, int(model.qk_rope_head_dim))
        op = FusedMlaAttnOp(
            name="fused_mla_attn",
            q_shape=(chunk, n_q, latent),
            kv_shape=(ctx, latent),
            dtype_bytes=model.dtype_bytes,
            kernel="prefill",
        )
        return float(op.features()["flops"])
    n_q, n_kv, d = _dense_heads(model)
    op = FusedAttnOp(
        name="fused_attn",
        q_shape=(chunk, n_q, d),
        k_shape=(ctx, n_kv, d),
        v_shape=(ctx, n_kv, d),
        dtype_bytes=model.dtype_bytes,
        kernel="prefill",
    )
    return float(op.features()["flops"])


def _qkv_proj_flops(*, tokens: int) -> float:
    model = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        intermediate_size=2816,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=64,
        attn_variant=AttnVariant.MHA,
    )
    h = model.hidden_size
    n_q, n_kv, d = _dense_heads(model)
    op = GemmOp(
        name="gemm_qkv",
        a_shape=(tokens, h),
        b_shape=(h, (n_q + 2 * n_kv) * d),
        dtype_bytes=model.dtype_bytes,
    )
    return float(op.features()["flops"])


class TestPrefillKvCacheCost(unittest.TestCase):
    def test_features_capture_cached_prefix(self) -> None:
        feats = extract_batch_features(
            _prefill_hit_batch(prompt=128, cached=64, chunk=32)
        )
        self.assertEqual(feats.phase, BatchPhase.PREFILL)
        self.assertEqual(feats.num_prefill_tokens, 32)
        self.assertEqual(feats.cached_prefix_tokens, 64)
        self.assertEqual(feats.cached_decode_tokens, 0)
        self.assertEqual(feats.prefill_chunk_lens, [32])
        self.assertEqual(feats.prefill_cached_lens, [64])

    def test_larger_cache_increases_prefill_attn_flops(self) -> None:
        cold = _attn_prefill_core_flops(chunk=32, cached=0)
        warm = _attn_prefill_core_flops(chunk=32, cached=96)
        self.assertGreater(warm, cold)
        self.assertAlmostEqual(warm / cold, 4.0, places=6)

    def test_multi_prefill_is_sum_not_product_of_sums(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            attn_variant=AttnVariant.MHA,
        )

        def fused_flops(feats: BatchFeatures) -> float:
            dag = build_operator_dag(
                model=model, parallel=ParallelConfig(), batch=feats
            )
            total = 0.0
            for op in dag.operators:
                if isinstance(op, FusedAttnOp):
                    total += float(lower_op(op).features.get("flops", 0.0))
            return total

        one = BatchFeatures(
            phase=BatchPhase.PREFILL,
            num_tokens=32,
            num_prefill_tokens=32,
            num_decode_tokens=0,
            batch_size=1,
            prefill_chunk_lens=[32],
            prefill_cached_lens=[0],
        )
        two = BatchFeatures(
            phase=BatchPhase.PREFILL,
            num_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=0,
            batch_size=2,
            prefill_chunk_lens=[32, 32],
            prefill_cached_lens=[0, 0],
        )
        self.assertAlmostEqual(fused_flops(two) / fused_flops(one), 2.0, places=6)

    def test_multi_decode_uses_per_request_kv(self) -> None:
        n_q, n_kv, d = 16, 16, 64
        a = FusedAttnOp(
            name="a",
            q_shape=(1, n_q, d),
            k_shape=(100, n_kv, d),
            v_shape=(100, n_kv, d),
            dtype_bytes=2,
            kernel="decode",
        )
        b = FusedAttnOp(
            name="b",
            q_shape=(3, n_q, d),
            k_shape=(1000, n_kv, d),
            v_shape=(1000, n_kv, d),
            dtype_bytes=2,
            kernel="decode",
        )
        flops = float(a.features()["flops"]) + float(b.features()["flops"])
        expected = 4.0 * 1 * n_q * d * 100 + 4.0 * 3 * n_q * d * 1000
        self.assertAlmostEqual(flops, expected, places=3)

    def test_smaller_chunk_reduces_linear_like_proj_cost(self) -> None:
        full = _qkv_proj_flops(tokens=64)
        hit = _qkv_proj_flops(tokens=32)
        self.assertAlmostEqual(hit / full, 0.5, places=6)

    def test_mla_prefill_uses_cached_prefix(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=2048,
            intermediate_size=4096,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=128,
            attn_variant=AttnVariant.MLA,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
        )
        cold = _attn_prefill_core_flops(chunk=16, cached=0, model=model)
        warm = _attn_prefill_core_flops(chunk=16, cached=48, model=model)
        self.assertGreater(warm, cold)
        self.assertAlmostEqual(warm / cold, (16 + 48) / 16, places=6)


if __name__ == "__main__":
    unittest.main()
