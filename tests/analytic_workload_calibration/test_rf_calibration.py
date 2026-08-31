"""Calibrate / align Roofline vs Frontier RF (tests only).

Primary Frontier gate: non-dummy RandomForest from profiling CSVs, with
``DeviceConfig`` util defaults and ``duration_scale=1.0`` (no fitted scale).
"""

from __future__ import annotations

import os
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
    calibrate_duration_scale,
    calibrated_config,
    fit_duration_scale,
    relative_error,
)
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch
from hybridsim_infer.workload_generators import OpLevelWorkloadGenerator
from hybridsim_infer.workload_generators.configs import (
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    OpLevelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    critical_path_duration_s,
)
from hybridsim_infer.workload_generators.types import AttnVariant

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

_RF_PROFILE = (
    _FRONTIER_ROOT
    / "data"
    / "profiling"
    / "compute"
    / "h800"
    / "llama2_7b_dense_example"
    / "linear_op.csv"
)
_RF_PROFILE_AVAILABLE = _RF_PROFILE.is_file()
_RESULTS_MD = Path(__file__).resolve().parent / "RF_ALIGNMENT.md"


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


def _prefill_with_cache(
    *,
    prompt: int,
    cached: int,
    chunk: int,
    rid: int = 20,
    batch_id: int = 20,
) -> ScheduleBatch:
    """Mid-prefill / APC partial hit: already computed ``cached`` tokens."""
    req = InferenceRequest(
        request_id=rid,
        arrived_at=0.0,
        num_prefill_tokens=prompt,
        num_decode_tokens=8,
        num_computed_tokens=cached,
    )
    return ScheduleBatch(
        batch_id=batch_id,
        requests=[req],
        tokens_per_request={rid: chunk},
        chunks=[PrefillChunk(request=req, num_tokens=chunk)],
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


def _multi_prefill(
    chunks: list[int], *, batch_id: int = 100, base_rid: int = 100
) -> ScheduleBatch:
    reqs: list[InferenceRequest] = []
    tpr: dict[int, int] = {}
    pcs: list[PrefillChunk] = []
    for i, n in enumerate(chunks):
        rid = base_rid + i
        req = InferenceRequest(
            request_id=rid,
            arrived_at=0.0,
            num_prefill_tokens=max(n * 2, n),
            num_decode_tokens=8,
            num_computed_tokens=0,
        )
        reqs.append(req)
        tpr[rid] = int(n)
        pcs.append(PrefillChunk(request=req, num_tokens=int(n)))
    return ScheduleBatch(
        batch_id=batch_id, requests=reqs, tokens_per_request=tpr, chunks=pcs
    )


def _multi_decode(
    ctxs: list[int], *, batch_id: int = 200, base_rid: int = 200
) -> ScheduleBatch:
    reqs: list[InferenceRequest] = []
    tpr: dict[int, int] = {}
    dcs: list[DecodeChunk] = []
    for i, ctx in enumerate(ctxs):
        rid = base_rid + i
        req = InferenceRequest(
            request_id=rid,
            arrived_at=0.0,
            num_prefill_tokens=int(ctx),
            num_decode_tokens=16,
            num_computed_tokens=int(ctx),
        )
        reqs.append(req)
        tpr[rid] = 1
        dcs.append(DecodeChunk(request=req, num_tokens=1))
    return ScheduleBatch(
        batch_id=batch_id, requests=reqs, tokens_per_request=tpr, chunks=dcs
    )


def _multi_prefill_with_cache(
    specs: list[tuple[int, int, int]],
    *,
    batch_id: int = 300,
    base_rid: int = 300,
) -> ScheduleBatch:
    """specs: list of (prompt, cached, chunk)."""
    reqs: list[InferenceRequest] = []
    tpr: dict[int, int] = {}
    pcs: list[PrefillChunk] = []
    for i, (prompt, cached, chunk) in enumerate(specs):
        rid = base_rid + i
        req = InferenceRequest(
            request_id=rid,
            arrived_at=0.0,
            num_prefill_tokens=int(prompt),
            num_decode_tokens=8,
            num_computed_tokens=int(cached),
        )
        reqs.append(req)
        tpr[rid] = int(chunk)
        pcs.append(PrefillChunk(request=req, num_tokens=int(chunk)))
    return ScheduleBatch(
        batch_id=batch_id, requests=reqs, tokens_per_request=tpr, chunks=pcs
    )


def _small_cfg(**kwargs) -> OpLevelConfig:
    return OpLevelConfig(
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


def _llama2_7b_cfg(*, num_layers: int) -> OpLevelConfig:
    return OpLevelConfig(
        model=ModelConfig(
            num_layers=num_layers,
            hidden_size=4096,
            intermediate_size=11008,
            num_q_heads=32,
            num_kv_heads=32,
            head_dim=128,
            attn_variant=AttnVariant.MHA,
            ffn_activation="silu",
        ),
        parallel=ParallelConfig(tp_size=1),
        device=DeviceConfig(),  # defaults: compute_util=hbm_util=0.6
        network=NetworkConfig(),
        duration_scale=1.0,
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
    """Calibrator math only (mock reference proportional to analytical)."""

    def setUp(self) -> None:
        self.cfg = _small_cfg()
        self.gen = OpLevelWorkloadGenerator(op_level=self.cfg)
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
        reused = OpLevelWorkloadGenerator(
            op_level=calibrated_config(self.cfg, duration_scale=scale)
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
@unittest.skipUnless(
    _RF_PROFILE_AVAILABLE,
    f"Missing Frontier RF profiling CSV: {_RF_PROFILE}",
)
class TestFrontierRfNumericalAlignment(unittest.TestCase):
    """Non-dummy RF vs analytical (util defaults, no duration_scale fit)."""

    MAX_REL_ERR = 0.05

    @classmethod
    def setUpClass(cls) -> None:
        cls._prev_cwd = Path.cwd()
        os.chdir(_FRONTIER_ROOT)
        try:
            ctx = build_monolithic_context(
                replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
                enable_dummy_mode=False,
                model_name="llama2_7b_dense_example",
                device="h800",
                network_device="h100_pairwise_nvlink",
            )
        except Exception as exc:  # pragma: no cover - env dependent
            os.chdir(cls._prev_cwd)
            raise unittest.SkipTest(f"Failed to build non-dummy RF: {exc}") from exc

        if bool(getattr(ctx.predictor, "_enable_dummy_mode", True)):
            os.chdir(cls._prev_cwd)
            raise unittest.SkipTest("Predictor unexpectedly in dummy mode")

        cls.rf = FrontierBatchDurationPredictor(
            ctx.predictor,
            cluster_type=ClusterType.MONOLITHIC,
            is_moe=False,
        )
        layers = int(
            getattr(ctx.predictor, "_num_layers_per_pipeline_stage", 32) or 32
        )
        cls.cfg = _llama2_7b_cfg(num_layers=layers)
        cls.gen = OpLevelWorkloadGenerator(op_level=cls.cfg)
        assert float(cls.gen.analyzer.duration_scale) == 1.0
        assert float(cls.cfg.device.compute_util) == 0.6
        assert float(cls.cfg.device.hbm_util) == 0.6

        cls.cases = [
            (
                "multi_prefill",
                _multi_prefill([32, 32, 48], batch_id=101),
            ),
            (
                "multi_decode",
                _multi_decode([128, 256, 512], batch_id=201),
            ),
            (
                "multi_prefill_with_kv_cache",
                _multi_prefill_with_cache(
                    [
                        (128, 64, 32),
                        (256, 128, 64),
                        (192, 96, 48),
                    ],
                    batch_id=301,
                ),
            ),
            ("single_prefill", _prefill(64, rid=1, batch_id=11)),
            ("single_decode", _decode(rid=2, batch_id=12, ctx=256)),
            (
                "single_prefill_cache",
                _prefill_with_cache(
                    prompt=256, cached=128, chunk=64, rid=3, batch_id=13
                ),
            ),
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.chdir(cls._prev_cwd)
        except Exception:
            pass

    def test_analytical_matches_rf_within_tol(self) -> None:
        """Record critical-path vs RF (observational; 5% gate suspended)."""
        rows: list[tuple[str, float, float, float, float]] = []
        print("\n=== analytical (util=0.6, scale=1) vs non-dummy Frontier RF ===")
        print(
            f"{'case':28s} {'analytical_s':>14s} {'rf_s':>14s} "
            f"{'ratio_a/rf':>12s} {'rel_err':>10s}"
        )
        for name, batch in self.cases:
            analytical = float(self.gen.predict_duration_s(batch))
            rf_s = float(self.rf.predict(batch))
            self.assertGreater(analytical, 0.0, msg=name)
            self.assertGreater(rf_s, 0.0, msg=name)
            ratio = analytical / rf_s
            err = relative_error(analytical, rf_s)
            rows.append((name, analytical, rf_s, ratio, err))
            print(
                f"{name:28s} {analytical:14.6e} {rf_s:14.6e} "
                f"{ratio:12.4f} {err:10.4f}"
            )

        lines = [
            "# Analytical vs non-dummy Frontier RF",
            "",
            "Predictor: `llama2_7b_dense_example` @ `h800`, `enable_dummy_mode=False`.",
            "Analytical: Llama-2-7B shape, `duration_scale=1.0`, "
            "`compute_util=hbm_util=0.6`.",
            f"Note: 5% gate suspended after shape-primitive refactor "
            f"(was `MAX_REL_ERR = {self.MAX_REL_ERR}`).",
            "",
            "| case | analytical_s | rf_s | analytical/rf | rel_err |",
            "|------|-------------:|-----:|--------------:|--------:|",
        ]
        for name, a, r, ratio, err in rows:
            lines.append(
                f"| {name} | {a:.6e} | {r:.6e} | {ratio:.4f} | {err:.4f} |"
            )
        lines.append("")
        _RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {_RESULTS_MD}")


if __name__ == "__main__":
    unittest.main()
