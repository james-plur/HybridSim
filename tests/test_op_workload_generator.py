"""Unit tests for analytical OpWorkloadGenerator / OpAnalyzer (RF-aligned)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch
from hybridsim_infer.workload_generators import (
    OpWorkloadGenerator,
    extract_batch_features,
    make_workload_generator,
)
from hybridsim_infer.workload_generators.analytic_model import (
    AnalyticalConfig,
    AttnVariant,
    BatchPhase,
    DeviceConfig,
    FfnActivation,
    ModelConfig,
    NetworkConfig,
    OperatorKind,
    ParallelConfig,
    TpCommStyle,
    build_operator_dag,
    critical_path_duration_s,
    expected_layer_op_names,
    strip_layer_prefix,
    total_kernel_duration_s,
)
from hybridsim_infer.workload_generators.analytic_model.models.ab_comm import (
    ab_comm_time_s,
)
from hybridsim_infer.workload_generators.analytic_model.models.roofline import (
    roofline_time_s,
)
from hybridsim_infer.workload_generators.analytic_model.operators.attention import (
    ensure_attn_variant_supported,
)
from hybridsim_infer.workload_generators.analytic_model.rf_catalog import (
    COMM_ATTN_TP_ALLREDUCE,
    COMM_EP_COMBINE,
    COMM_EP_DISPATCH,
    COMM_MLP_TP_ALLREDUCE,
    COMM_PP_SEND_RECV,
)


def _prefill_batch(tokens: int = 64) -> ScheduleBatch:
    req = InferenceRequest(
        request_id=1,
        arrived_at=0.0,
        num_prefill_tokens=128,
        num_decode_tokens=8,
        num_computed_tokens=0,
    )
    return ScheduleBatch(
        batch_id=1,
        requests=[req],
        tokens_per_request={1: tokens},
        chunks=[PrefillChunk(request=req, num_tokens=tokens)],
    )


def _decode_batch() -> ScheduleBatch:
    req = InferenceRequest(
        request_id=2,
        arrived_at=0.0,
        num_prefill_tokens=128,
        num_decode_tokens=16,
        num_computed_tokens=128,
    )
    return ScheduleBatch(
        batch_id=2,
        requests=[req],
        tokens_per_request={2: 1},
        chunks=[DecodeChunk(request=req, num_tokens=1)],
    )


def _validate_dag(kernels: list[dict]) -> None:
    n = len(kernels)
    in_degree = [0] * n
    for i, k in enumerate(kernels):
        for d in k["dependencies"]:
            assert 0 <= d < n
            assert d != i
            in_degree[i] += 1
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for i, k in enumerate(kernels):
            if node in k["dependencies"]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
    assert visited == n, "kernel DAG contains a cycle"


class TestBatchFeatures(unittest.TestCase):
    def test_prefill_phase(self) -> None:
        feats = extract_batch_features(_prefill_batch(32))
        self.assertEqual(feats.phase, BatchPhase.PREFILL)
        self.assertEqual(feats.num_prefill_tokens, 32)

    def test_decode_phase(self) -> None:
        feats = extract_batch_features(_decode_batch())
        self.assertEqual(feats.phase, BatchPhase.DECODE)
        self.assertGreater(feats.kv_cache_tokens, 0)


class TestRooflineAndAB(unittest.TestCase):
    def test_roofline_compute_bound(self) -> None:
        device = DeviceConfig(peak_flops=100.0, hbm_bandwidth_bps=1e12)
        self.assertAlmostEqual(
            roofline_time_s(flops=1000.0, bytes_=1.0, device=device), 10.0
        )

    def test_ab_zero_when_single_rank(self) -> None:
        net = NetworkConfig(alpha_s=1e-6, beta_s_per_byte=1e-9)
        self.assertEqual(
            ab_comm_time_s(
                payload_bytes=1e6, volume_factor=1.0, network=net, num_ranks=1
            ),
            0.0,
        )


class TestRfAlignedDAG(unittest.TestCase):
    def test_dense_prefill_layer_matches_catalog(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(num_layers=1, attn_variant=AttnVariant.GQA),
            parallel=ParallelConfig(),
        )
        feats = extract_batch_features(_prefill_batch())
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        got = dag.rf_op_names()
        expect = expected_layer_op_names(
            attn_variant=AttnVariant.GQA,
            phase=BatchPhase.PREFILL,
            is_moe=False,
        )
        self.assertEqual(got, expect)

    def test_dense_tp_inserts_frontier_allreduce_names(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(num_layers=1),
            parallel=ParallelConfig(tp_size=4),
        )
        feats = extract_batch_features(_prefill_batch())
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        names = dag.rf_op_names()
        self.assertIn(COMM_ATTN_TP_ALLREDUCE, names)
        self.assertIn(COMM_MLP_TP_ALLREDUCE, names)

    def test_pp_uses_frontier_send_recv_name(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(num_layers=4),
            parallel=ParallelConfig(pp_size=2, pp_stage=0),
        )
        feats = extract_batch_features(_prefill_batch())
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        self.assertIn(COMM_PP_SEND_RECV, dag.rf_op_names())

    def test_moe_ep_matches_catalog(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(
                num_layers=1,
                is_moe=True,
                num_experts=8,
                num_experts_per_tok=2,
                share_expert_dim=2048,
            ),
            parallel=ParallelConfig(tp_size=2, ep_size=4, moe_tp_size=2),
        )
        feats = extract_batch_features(_prefill_batch())
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        expect = expected_layer_op_names(
            attn_variant=cfg.model.resolved_attn_variant(),
            phase=BatchPhase.PREFILL,
            is_moe=True,
            has_share_expert=True,
            attn_tp=2,
            moe_tp=2,
            ep=4,
        )
        self.assertEqual(dag.rf_op_names(), expect)
        self.assertIn(COMM_EP_DISPATCH, dag.rf_op_names())
        self.assertIn(COMM_EP_COMBINE, dag.rf_op_names())

    def test_mla_prefill_ops(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(num_layers=1, attn_variant=AttnVariant.MLA),
            parallel=ParallelConfig(),
        )
        feats = extract_batch_features(_prefill_batch())
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        expect = expected_layer_op_names(
            attn_variant=AttnVariant.MLA,
            phase=BatchPhase.PREFILL,
        )
        self.assertEqual(dag.rf_op_names(), expect)

    def test_legacy_rs_ag_style_still_builds(self) -> None:
        # RS/AG remains available via ParallelConfig; default dag uses allreduce.
        cfg = AnalyticalConfig(
            model=ModelConfig(num_layers=1),
            parallel=ParallelConfig(tp_size=2, tp_comm_style=TpCommStyle.RS_AG),
        )
        feats = extract_batch_features(_prefill_batch())
        # RF-aligned builder always uses allreduce for TP; style reserved for legacy helper.
        dag = build_operator_dag(model=cfg.model, parallel=cfg.parallel, batch=feats)
        self.assertIn(COMM_ATTN_TP_ALLREDUCE, dag.rf_op_names())


class TestOpAnalyzerExpand(unittest.TestCase):
    def test_ffn_ops_are_separate_kernels(self) -> None:
        gen = OpWorkloadGenerator(
            analytical=AnalyticalConfig(model=ModelConfig(num_layers=1)),
        )
        wl = gen(_prefill_batch(), workload_id=9)
        _validate_dag(wl["kernels"])
        names = [strip_layer_prefix(k["name"]) for k in wl["kernels"]]
        for op in ("mlp_up_proj", "mlp_act", "mlp_down_proj"):
            self.assertIn(op, names)
        up = next(k for k in wl["kernels"] if k["name"].endswith("mlp_up_proj"))
        act = next(k for k in wl["kernels"] if k["name"].endswith("mlp_act"))
        down = next(k for k in wl["kernels"] if k["name"].endswith("mlp_down_proj"))
        self.assertIn(wl["kernels"].index(up), act["dependencies"])
        self.assertIn(wl["kernels"].index(act), down["dependencies"])

    def test_durations_positive(self) -> None:
        wl = make_workload_generator(duration_mode="analytical")(
            _prefill_batch(), workload_id=1
        )
        self.assertTrue(all(k["duration"] >= 0.0 for k in wl["kernels"]))
        self.assertGreater(critical_path_duration_s(wl["kernels"]), 0.0)
        self.assertGreater(total_kernel_duration_s(wl["kernels"]), 0.0)


class TestVariants(unittest.TestCase):
    def test_mha_gqa_mqa_mla_build(self) -> None:
        for variant in (
            AttnVariant.MHA,
            AttnVariant.GQA,
            AttnVariant.MQA,
            AttnVariant.MLA,
        ):
            cfg = AnalyticalConfig(
                model=ModelConfig(num_layers=1, attn_variant=variant)
            )
            wl = OpWorkloadGenerator(analytical=cfg)(_prefill_batch(), workload_id=1)
            self.assertGreater(len(wl["kernels"]), 0)

    def test_ffn_activations(self) -> None:
        for act in (
            FfnActivation.GELU,
            FfnActivation.SILU,
            FfnActivation.SWIGLU,
            FfnActivation.RELU,
        ):
            cfg = AnalyticalConfig(
                model=ModelConfig(num_layers=1, ffn_activation=act)
            )
            wl = OpWorkloadGenerator(analytical=cfg)(_prefill_batch(), workload_id=1)
            self.assertGreater(len(wl["kernels"]), 0)

    def test_stub_variants_raise(self) -> None:
        for variant in (AttnVariant.CSA, AttnVariant.HSA, AttnVariant.DSA):
            with self.assertRaises(NotImplementedError):
                ensure_attn_variant_supported(variant)


class TestFactoryAnalytical(unittest.TestCase):
    def test_factory_mode_aliases(self) -> None:
        for mode in ("analytical", "analytic", "op", "kernel_dag"):
            gen = make_workload_generator(duration_mode=mode)
            self.assertIsInstance(gen, OpWorkloadGenerator)


if __name__ == "__main__":
    unittest.main()
