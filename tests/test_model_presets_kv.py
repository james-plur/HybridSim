"""Tests for model presets, KV volume formulas, DSA DAG, and KV transfer α-β."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.kv_system.client import KvClient
from hybridsim_infer.workload_generators.analytic_model import (
    AttnVariant,
    BatchFeatures,
    BatchPhase,
    ModelConfig,
    NetworkConfig,
    ParallelConfig,
    build_operator_dag,
    bytes_per_token,
    cache_bytes,
    list_presets,
    load_preset,
)
from hybridsim_infer.workload_generators.analytic_model.model_presets import (
    preset_meta,
)
from hybridsim_infer.workload_generators.kv_transfer import (
    KvTransferWorkloadGenerator,
    transfer_duration_s,
)

_EXPECTED_IDS = {
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v3.2",
    "deepseek-v3",
    "deepseek-r1",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.3-70b",
}


class TestModelPresets(unittest.TestCase):
    def test_list_all_four_families(self) -> None:
        ids = set(list_presets())
        self.assertEqual(ids, _EXPECTED_IDS)
        self.assertEqual(len(list_presets(family="deepseek")), 5)
        self.assertEqual(len(list_presets(family="glm")), 3)
        self.assertEqual(len(list_presets(family="kimi")), 2)
        self.assertEqual(len(list_presets(family="llama")), 3)

    def test_load_preset_fields_match_yaml(self) -> None:
        for pid in sorted(_EXPECTED_IDS):
            model = load_preset(pid)
            meta = preset_meta(pid)
            self.assertEqual(model.num_layers, meta["num_layers"])
            self.assertEqual(model.kv_formula, meta["kv_formula"])
            self.assertEqual(
                model.resolved_attn_variant().value, meta["attn_variant"]
            )
            self.assertEqual(model.dtype_bytes, 2)
            self.assertEqual(model.hidden_size, meta["hidden_size"])
            self.assertEqual(model.num_kv_heads, meta["num_kv_heads"])


class TestKvCacheBytes(unittest.TestCase):
    """Hand-calculated against elinx calc.js formulas (BF16, T=1024, no draft)."""

    T = 1024
    P = 2

    def test_llama_3_1_8b_gqa(self) -> None:
        m = load_preset("llama-3.1-8b")
        # L × 2 × n_kv × hd × T × p
        expected = 32 * 2 * 8 * 128 * self.T * self.P
        self.assertEqual(cache_bytes(m, self.T), float(expected))
        self.assertEqual(bytes_per_token(m, num_tokens=1), float(2 * 8 * 128 * self.P * 32))

    def test_deepseek_v3_mla(self) -> None:
        m = load_preset("deepseek-v3")
        # L × (kv_lora + rope) × T × p
        expected = 61 * (512 + 64) * self.T * self.P
        self.assertEqual(cache_bytes(m, self.T), float(expected))

    def test_kimi_k2_6_mla(self) -> None:
        m = load_preset("kimi-k2.6")
        expected = 61 * (512 + 64) * self.T * self.P
        self.assertEqual(cache_bytes(m, self.T), float(expected))

    def test_glm_5_dsa_mla(self) -> None:
        m = load_preset("glm-5")
        kv = 78 * (512 + 64) * self.T * self.P
        idx = 78 * 128 * self.T * self.P
        self.assertEqual(cache_bytes(m, self.T), float(kv + idx))

    def test_glm_5_2_index_share(self) -> None:
        m = load_preset("glm-5.2")
        # IndexShare: offset=3, freq=4 → 3 + floor((78-3)/4) = 3+18 = 21
        n_idx = 3 + (78 - 3) // 4
        self.assertEqual(n_idx, 21)
        kv = 78 * (512 + 64) * self.T * self.P
        idx = n_idx * 128 * self.T * self.P
        self.assertEqual(cache_bytes(m, self.T), float(kv + idx))


class TestDsaDag(unittest.TestCase):
    def test_dsa_preset_builds_dag(self) -> None:
        model = load_preset("glm-5")
        # Shrink layers for a fast structural check.
        model.num_layers = 2
        batch = BatchFeatures(
            phase=BatchPhase.PREFILL,
            num_tokens=16,
            num_prefill_tokens=16,
            num_decode_tokens=0,
            batch_size=1,
            cached_decode_tokens=0,
        )
        dag = build_operator_dag(
            model=model,
            parallel=ParallelConfig(),
            batch=batch,
        )
        names = [op.name for op in dag.operators]
        self.assertTrue(any("attn_mla_prefill" in n for n in names))
        self.assertTrue(any("attn_indexer_cache_save" in n for n in names))

    def test_deepseek_v32_dsa(self) -> None:
        model = load_preset("deepseek-v3.2")
        self.assertEqual(model.resolved_attn_variant(), AttnVariant.DSA)
        model.num_layers = 1
        batch = BatchFeatures(
            phase=BatchPhase.DECODE,
            num_tokens=1,
            num_prefill_tokens=0,
            num_decode_tokens=1,
            batch_size=1,
            cached_decode_tokens=128,
        )
        dag = build_operator_dag(
            model=model, parallel=ParallelConfig(), batch=batch
        )
        self.assertGreater(len(dag.operators), 0)


class TestKvTransfer(unittest.TestCase):
    def test_alpha_beta_duration(self) -> None:
        model = load_preset("llama-3.1-8b")
        tokens = 1024
        nbytes = cache_bytes(model, tokens)
        bw_gbps = 100.0
        alpha = 1e-3
        bps = bw_gbps * 1e9 / 8.0
        expected = alpha + nbytes / bps
        got = transfer_duration_s(
            num_tokens=tokens,
            model=model,
            bandwidth_gbps=bw_gbps,
            latency_s=alpha,
        )
        self.assertAlmostEqual(got, expected, places=12)

    def test_workload_generator_dict(self) -> None:
        model = load_preset("deepseek-v3")
        net = NetworkConfig.from_bandwidth(latency_s=2e-4, bandwidth_gbps=50.0)
        gen = KvTransferWorkloadGenerator(model=model, network=net)
        wl = gen(
            workload_id=7,
            request_id=3,
            direction="pull",
            num_tokens=512,
        )
        self.assertEqual(wl["workload_id"], 7)
        self.assertEqual(len(wl["kernels"]), 1)
        self.assertAlmostEqual(
            wl["kernels"][0]["duration"],
            gen.estimate_duration_s(512),
            places=12,
        )

    def test_page_mode_serial_kernels(self) -> None:
        model = ModelConfig(
            num_layers=2,
            num_kv_heads=1,
            head_dim=64,
            kv_formula="standard_gqa",
            dtype_bytes=2,
        )
        gen = KvTransferWorkloadGenerator(
            model=model,
            bandwidth_gbps=50.0,
            latency_s=1e-3,
            page_tokens=256,
        )
        wl = gen(workload_id=1, request_id=1, num_tokens=512, direction="push")
        self.assertEqual(len(wl["kernels"]), 2)
        self.assertEqual(wl["kernels"][1]["dependencies"], [0])

    def test_kv_client_uses_model_bytes(self) -> None:
        model = load_preset("llama-3.1-8b")
        engine = MagicMock()
        owner = MagicMock()
        owner.sim.now.return_value = 0.0
        client = KvClient(
            owner,
            store=None,
            engine=engine,
            bandwidth_gbps=50.0,
            bytes_per_token=16.0,
            transfer_s_floor=0.0,
            kv_latency_s=1e-3,
            model_config=model,
            on_transfer_complete=lambda *a: None,
        )
        tokens = 256
        expected = transfer_duration_s(
            num_tokens=tokens,
            model=model,
            bandwidth_gbps=50.0,
            latency_s=1e-3,
        )
        self.assertAlmostEqual(client.transfer_duration_s(tokens), expected, places=12)
        # Model-derived bytes/token differs from the 16-byte fallback.
        self.assertNotEqual(client.bytes_per_token, 16.0)


if __name__ == "__main__":
    unittest.main()
