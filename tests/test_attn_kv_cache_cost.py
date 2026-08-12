"""Prefill attention cost respects cached prefix / KV context (Frontier-aligned)."""

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
from hybridsim_infer.workload_generators.analytic_model import (
    AttnVariant,
    BatchFeatures,
    BatchPhase,
    ModelConfig,
)
from hybridsim_infer.workload_generators.analytic_model.operators.attention import (
    make_attn_block_operators,
)


def _prefill_batch_features(*, chunk: int, cached: int) -> BatchFeatures:
    return BatchFeatures(
        phase=BatchPhase.PREFILL,
        num_tokens=chunk,
        num_prefill_tokens=chunk,
        num_decode_tokens=0,
        batch_size=1,
        cached_decode_tokens=0,
        cached_prefix_tokens=cached,
        prefill_chunk_lens=[chunk],
        prefill_cached_lens=[cached],
    )


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
    batch = _prefill_batch_features(chunk=chunk, cached=cached)
    ops = make_attn_block_operators(
        layer_id=0, model=model, batch=batch, deps=[], tp_size=1
    )
    for op in ops:
        if op.features.get("rf_op") in ("attn_prefill", "attn_mla_prefill"):
            return float(op.features["flops"])
    raise AssertionError("prefill attention op not found")


def _pre_proj_flops(*, tokens: int, cached: int = 64) -> float:
    model = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        intermediate_size=2816,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=64,
        attn_variant=AttnVariant.MHA,
    )
    batch = _prefill_batch_features(chunk=tokens, cached=cached)
    ops = make_attn_block_operators(
        layer_id=0, model=model, batch=batch, deps=[], tp_size=1
    )
    for op in ops:
        if op.features.get("rf_op") == "attn_pre_proj":
            return float(op.features["flops"])
    raise AssertionError("attn_pre_proj not found")


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


class TestPrefillKvCacheCost(unittest.TestCase):
    def test_features_capture_cached_prefix(self) -> None:
        feats = extract_batch_features(
            _prefill_hit_batch(prompt=128, cached=64, chunk=32)
        )
        self.assertEqual(feats.phase, BatchPhase.PREFILL)
        self.assertEqual(feats.num_prefill_tokens, 32)
        self.assertEqual(feats.cached_prefix_tokens, 64)
        self.assertEqual(feats.cached_decode_tokens, 0)  # decode-only aggregate
        self.assertEqual(feats.prefill_chunk_lens, [32])
        self.assertEqual(feats.prefill_cached_lens, [64])

    def test_larger_cache_increases_prefill_attn_flops(self) -> None:
        cold = _attn_prefill_core_flops(chunk=32, cached=0)
        warm = _attn_prefill_core_flops(chunk=32, cached=96)
        self.assertGreater(warm, cold)
        # chunk×ctx vs chunk×chunk → (32+96)/32 = 4×
        self.assertAlmostEqual(warm / cold, 4.0, places=6)

    def test_multi_prefill_is_sum_not_product_of_sums(self) -> None:
        """N identical prefills must cost N× one request, not N²."""
        model = ModelConfig(
            num_layers=1,
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            attn_variant=AttnVariant.MHA,
        )
        one = BatchFeatures(
            phase=BatchPhase.PREFILL,
            num_tokens=32,
            num_prefill_tokens=32,
            num_decode_tokens=0,
            batch_size=1,
            cached_prefix_tokens=0,
            prefill_chunk_lens=[32],
            prefill_cached_lens=[0],
        )
        two = BatchFeatures(
            phase=BatchPhase.PREFILL,
            num_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=0,
            batch_size=2,
            cached_prefix_tokens=0,
            prefill_chunk_lens=[32, 32],
            prefill_cached_lens=[0, 0],
        )

        def prefill_flops(feats: BatchFeatures) -> float:
            ops = make_attn_block_operators(
                layer_id=0, model=model, batch=feats, deps=[], tp_size=1
            )
            for op in ops:
                if op.features.get("rf_op") == "attn_prefill":
                    return float(op.features["flops"])
            raise AssertionError("missing attn_prefill")

        self.assertAlmostEqual(prefill_flops(two) / prefill_flops(one), 2.0, places=6)

    def test_multi_decode_uses_per_request_kv(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            attn_variant=AttnVariant.MHA,
        )
        # 1 tok @ ctx 100 + 3 tok @ ctx 1000 → 100 + 3000 = 3100 scale units
        uneven = BatchFeatures(
            phase=BatchPhase.DECODE,
            num_tokens=4,
            num_prefill_tokens=0,
            num_decode_tokens=4,
            batch_size=2,
            cached_decode_tokens=1100,
            decode_token_lens=[1, 3],
            decode_kv_lens=[100, 1000],
        )
        # Mean approximation would be wrong; per-request sum is reference.
        ops = make_attn_block_operators(
            layer_id=0, model=model, batch=uneven, deps=[], tp_size=1
        )
        flops = next(
            float(o.features["flops"])
            for o in ops
            if o.features.get("rf_op") == "attn_decode"
        )
        n_q, _, d = 16, 16, 64
        expected = 4.0 * 1 * n_q * d * 100 + 4.0 * 3 * n_q * d * 1000
        self.assertAlmostEqual(flops, expected, places=3)

    def test_smaller_chunk_reduces_linear_like_proj_cost(self) -> None:
        full = _pre_proj_flops(tokens=64)
        hit = _pre_proj_flops(tokens=32)  # 50% remaining after cache hit
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
