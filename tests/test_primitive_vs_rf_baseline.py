"""Block-level flops/bytes: shape primitives vs frozen rf_op costing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
_TESTS = _HYBRIDSIM_ROOT / "tests"
for _p in (_PY, _TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
    BatchPhase,
)
from rf_baseline.rf_block_cost import rf_block_cost, sum_dag_blocks
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    build_operator_dag,
)
from hybridsim_infer.workload_generators.types import AttnVariant, FfnActivation


def _prefill(*, tokens: int = 32, cached: int = 0) -> BatchFeatures:
    return BatchFeatures(
        phase=BatchPhase.PREFILL,
        num_tokens=tokens,
        num_prefill_tokens=tokens,
        num_decode_tokens=0,
        batch_size=1,
        cached_prefix_tokens=cached,
        prefill_chunk_lens=[tokens],
        prefill_cached_lens=[cached],
    )


def _decode(*, tokens: int = 1, ctx: int = 128) -> BatchFeatures:
    return BatchFeatures(
        phase=BatchPhase.DECODE,
        num_tokens=tokens,
        num_prefill_tokens=0,
        num_decode_tokens=tokens,
        batch_size=1,
        cached_decode_tokens=ctx,
        decode_token_lens=[tokens],
        decode_kv_lens=[ctx],
    )


class TestPrimitiveVsRfBaseline(unittest.TestCase):
    def test_dense_prefill_core_and_ffn_match(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            attn_variant=AttnVariant.MHA,
            ffn_activation=FfnActivation.SILU,
        )
        parallel = ParallelConfig()
        batch = _prefill(tokens=32, cached=16)
        dag = build_operator_dag(model=model, parallel=parallel, batch=batch)
        got = sum_dag_blocks(dag)
        exp = rf_block_cost(model=model, parallel=parallel, batch=batch)

        self.assertAlmostEqual(got["attn_core"]["flops"], exp["attn_core"]["flops"])
        self.assertAlmostEqual(got["attn_core"]["bytes"], exp["attn_core"]["bytes"])
        self.assertAlmostEqual(got["ffn"]["flops"], exp["ffn"]["flops"])
        self.assertAlmostEqual(got["ffn"]["bytes"], exp["ffn"]["bytes"])
        self.assertAlmostEqual(got["layer_mem"]["bytes"], exp["layer_mem"]["bytes"])
        self.assertAlmostEqual(got["attn_side"]["bytes"], exp["attn_side"]["bytes"])

        self.assertAlmostEqual(got["attn_proj"]["flops"], exp["attn_proj"]["flops"])
        self.assertAlmostEqual(got["attn_proj"]["bytes"], exp["attn_proj"]["bytes"])

    def test_dense_decode_core_match(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=64,
            attn_variant=AttnVariant.GQA,
        )
        parallel = ParallelConfig()
        batch = _decode(tokens=1, ctx=256)
        dag = build_operator_dag(model=model, parallel=parallel, batch=batch)
        got = sum_dag_blocks(dag)
        exp = rf_block_cost(model=model, parallel=parallel, batch=batch)
        self.assertAlmostEqual(got["attn_core"]["flops"], exp["attn_core"]["flops"])
        self.assertAlmostEqual(got["attn_core"]["bytes"], exp["attn_core"]["bytes"])

    def test_tp_comm_volume_match(self) -> None:
        model = ModelConfig(num_layers=1, attn_variant=AttnVariant.GQA)
        parallel = ParallelConfig(tp_size=4)
        batch = _prefill()
        dag = build_operator_dag(model=model, parallel=parallel, batch=batch)
        got = sum_dag_blocks(dag)
        exp = rf_block_cost(model=model, parallel=parallel, batch=batch)
        self.assertAlmostEqual(got["comm"]["bytes"], exp["comm"]["bytes"])
        self.assertGreater(got["comm"]["bytes"], 0.0)

    def test_mla_core_match(self) -> None:
        model = ModelConfig(
            num_layers=1,
            hidden_size=2048,
            intermediate_size=4096,
            num_q_heads=16,
            attn_variant=AttnVariant.MLA,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
        )
        parallel = ParallelConfig()
        batch = _prefill(tokens=16, cached=48)
        dag = build_operator_dag(model=model, parallel=parallel, batch=batch)
        got = sum_dag_blocks(dag)
        exp = rf_block_cost(model=model, parallel=parallel, batch=batch)
        self.assertAlmostEqual(got["attn_core"]["flops"], exp["attn_core"]["flops"])
        self.assertAlmostEqual(got["attn_core"]["bytes"], exp["attn_core"]["bytes"])

    def test_moe_gate_and_grouped_flops(self) -> None:
        model = ModelConfig(
            num_layers=1,
            is_moe=True,
            num_experts=8,
            num_experts_per_tok=2,
            share_expert_dim=0,
            attn_variant=AttnVariant.GQA,
        )
        parallel = ParallelConfig()
        batch = _prefill(tokens=16)
        dag = build_operator_dag(model=model, parallel=parallel, batch=batch)
        got = sum_dag_blocks(dag)
        exp = rf_block_cost(model=model, parallel=parallel, batch=batch)
        self.assertAlmostEqual(got["moe"]["flops"], exp["moe"]["flops"])
        # Two GEMMs write a 2*i_local up-activation; rf grouped_gemm counted i_local once.
        s = batch.num_tokens
        k = model.num_experts_per_tok
        i_local = max(1, int(model.intermediate_size))
        extra = float(2 * s * k * i_local * model.dtype_bytes)
        self.assertAlmostEqual(got["moe"]["bytes"], exp["moe"]["bytes"] + extra)


if __name__ == "__main__":
    unittest.main()
