"""Calibrate Roofline critical-path against RF / mock reference (tests only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[2]
_FRONTIER_EXAMPLE = _HYBRIDSIM_ROOT / "examples" / "frontier"
_FRONTIER_ROOT = Path("/home/y_luchenda/Frontier")
_PY = _HYBRIDSIM_ROOT / "src" / "python"
_TESTS = _HYBRIDSIM_ROOT / "tests"

for _p in (_PY, _TESTS, _FRONTIER_EXAMPLE, _FRONTIER_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from analytic_workload_calibration.calibrator import (
    alignment_errors,
    calibrate_duration_scale,
    calibrated_config,
    fit_duration_scale,
    relative_error,
)
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch
from hybridsim_infer.workload_generators import OpWorkloadGenerator
from hybridsim_infer.workload_generators.analytic_model import (
    AnalyticalConfig,
    AttnVariant,
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    ParallelConfig,
    critical_path_duration_s,
)

try:
    from frontier.types import ClusterType
    from frontier_bridge.context import (
        ReplicaSchedulerKind,
        build_monolithic_context,
    )
    from hybridsim_infer.workload_generators import FrontierBatchDurationPredictor

    _FRONTIER_AVAILABLE = True
except ImportError:
    _FRONTIER_AVAILABLE = False


def _prefill(tokens: int, *, rid: int = 1, batch_id: int = 1) -> ScheduleBatch:
    req = InferenceRequest(
        request_id=rid,
        arrived_at=0.0,
        num_prefill_tokens=max(tokens * 2, tokens),
        num_decode_tokens=8,
        num_computed_tokens=0,
    )
    return ScheduleBatch(
        batch_id=batch_id,
        requests=[req],
        tokens_per_request={rid: tokens},
        chunks=[PrefillChunk(request=req, num_tokens=tokens)],
    )


def _decode(*, rid: int = 2, batch_id: int = 2, ctx: int = 128) -> ScheduleBatch:
    req = InferenceRequest(
        request_id=rid,
        arrived_at=0.0,
        num_prefill_tokens=ctx,
        num_decode_tokens=16,
        num_computed_tokens=ctx,
    )
    return ScheduleBatch(
        batch_id=batch_id,
        requests=[req],
        tokens_per_request={rid: 1},
        chunks=[DecodeChunk(request=req, num_tokens=1)],
    )


def _small_cfg(**kwargs) -> AnalyticalConfig:
    return AnalyticalConfig(
        model=ModelConfig(
            num_layers=kwargs.get("num_layers", 4),
            hidden_size=1024,
            intermediate_size=2816,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            attn_variant=AttnVariant.MHA,
        ),
        parallel=ParallelConfig(tp_size=1),
        device=DeviceConfig(),
        network=NetworkConfig(),
    )


class TestFitDurationScale(unittest.TestCase):
    def test_least_squares_recovers_constant(self) -> None:
        self.assertAlmostEqual(
            fit_duration_scale([1.0, 2.0, 4.0], [3.0, 6.0, 12.0]), 3.0
        )

    def test_rejects_all_zero_analytical(self) -> None:
        with self.assertRaises(ValueError):
            fit_duration_scale([0.0, 0.0], [1.0, 2.0])


class TestMockRfNumericalAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _small_cfg()
        self.gen = OpWorkloadGenerator(analytical=self.cfg)
        self.batches = [
            _prefill(16, rid=1, batch_id=1),
            _prefill(32, rid=2, batch_id=2),
            _prefill(64, rid=3, batch_id=3),
            _decode(rid=4, batch_id=4, ctx=64),
        ]
        self.k = 2.5

        def mock_rf(batch: ScheduleBatch) -> float:
            old = self.gen.analyzer.duration_scale
            self.gen.analyzer.duration_scale = 1.0
            raw = self.gen.predict_duration_s(batch)
            self.gen.analyzer.duration_scale = old
            return self.k * raw

        self.mock_rf = mock_rf

    def test_calibrate_recovers_scale_and_aligns_holdout(self) -> None:
        train, holdout = self.batches[:2], self.batches[2:]
        scale = calibrate_duration_scale(self.gen, train, self.mock_rf)
        self.assertAlmostEqual(scale, self.k, places=6)

        for batch in train + holdout:
            err = relative_error(
                self.gen.predict_duration_s(batch), self.mock_rf(batch)
            )
            self.assertLessEqual(err, 1e-6)

    def test_reuse_calibrated_config(self) -> None:
        scale = calibrate_duration_scale(self.gen, self.batches[:1], self.mock_rf)
        reused = OpWorkloadGenerator(
            analytical=calibrated_config(self.cfg, duration_scale=scale)
        )
        batch = self.batches[0]
        self.assertAlmostEqual(
            reused.predict_duration_s(batch),
            self.mock_rf(batch),
            places=9,
        )
        wl = reused(batch, workload_id=1)
        self.assertAlmostEqual(
            critical_path_duration_s(wl["kernels"]),
            self.mock_rf(batch),
            places=9,
        )


@unittest.skipUnless(_FRONTIER_AVAILABLE, "Frontier not installed / not on PYTHONPATH")
class TestFrontierRfNumericalAlignment(unittest.TestCase):
    MAX_REL_ERR = 0.05

    def setUp(self) -> None:
        ctx = build_monolithic_context(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=10.0,
            model_name="meta-llama/Llama-2-7b-hf",
        )
        self.rf = FrontierBatchDurationPredictor(
            ctx.predictor,
            cluster_type=ClusterType.MONOLITHIC,
            is_moe=False,
        )
        layers = int(
            getattr(ctx.predictor, "_num_layers_per_pipeline_stage", 32) or 32
        )
        self.cfg = AnalyticalConfig(
            model=ModelConfig(
                num_layers=layers,
                hidden_size=4096,
                intermediate_size=11008,
                num_q_heads=32,
                num_kv_heads=32,
                head_dim=128,
                attn_variant=AttnVariant.MHA,
                ffn_activation="silu",
            ),
            parallel=ParallelConfig(tp_size=1),
            device=DeviceConfig(),
            network=NetworkConfig(),
        )
        self.gen = OpWorkloadGenerator(analytical=self.cfg)
        self.batches = [
            _prefill(32, rid=1, batch_id=10),
            _prefill(64, rid=2, batch_id=11),
            _decode(rid=3, batch_id=12, ctx=128),
        ]

    def test_calibrate_then_match_rf_total_time(self) -> None:
        scale = calibrate_duration_scale(self.gen, self.batches, self.rf.predict)
        self.assertGreater(scale, 0.0)
        errs = alignment_errors(self.gen, self.batches, self.rf.predict)
        for batch, err in zip(self.batches, errs):
            got = self.gen.predict_duration_s(batch)
            ref = float(self.rf.predict(batch))
            print(
                f"rf-align batch={batch.batch_id}: "
                f"analytical={got:.6e}s rf={ref:.6e}s "
                f"scale={scale:.6e} rel_err={err:.4f}"
            )
            self.assertLessEqual(err, self.MAX_REL_ERR)
        self.assertLessEqual(sum(errs) / len(errs), self.MAX_REL_ERR)

        # Reuse calibrated params on a fresh generator (production path).
        reused = OpWorkloadGenerator(
            analytical=calibrated_config(self.cfg, duration_scale=scale)
        )
        for batch in self.batches:
            self.assertLessEqual(
                relative_error(
                    reused.predict_duration_s(batch), float(self.rf.predict(batch))
                ),
                self.MAX_REL_ERR,
            )

    def test_uncalibrated_differs_then_calibrated_matches(self) -> None:
        batch = self.batches[0]
        raw = self.gen.predict_duration_s(batch)
        rf_s = float(self.rf.predict(batch))
        self.assertGreater(relative_error(raw, rf_s), 0.5)
        calibrate_duration_scale(self.gen, [batch], self.rf.predict)
        self.assertLessEqual(
            relative_error(self.gen.predict_duration_s(batch), rf_s),
            self.MAX_REL_ERR,
        )


if __name__ == "__main__":
    unittest.main()
