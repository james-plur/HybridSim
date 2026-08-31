"""Align BatchLevelWorkloadGenerator with Frontier BaseExecutionTimePredictor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_FRONTIER_EXAMPLE = _HYBRIDSIM_ROOT / "examples" / "frontier"
_FRONTIER_ROOT = Path("/home/y_luchenda/Frontier")

for _p in (_HYBRIDSIM_ROOT / "src" / "python", _FRONTIER_EXAMPLE, _FRONTIER_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    import frontier  # noqa: F401

    from frontier.types import ClusterType
    from frontier_bridge.batch_executor import predict_batch_duration
    from frontier_bridge.context import (
        ReplicaSchedulerKind,
        build_monolithic_context,
    )
    from hybridsim_infer.request import InferenceRequest
    from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch
    from hybridsim_infer.workload_generators import (
        BatchLevelWorkloadGenerator,
        FrontierBatchDurationPredictor,
        make_infer_workload_generator,
    )

    _FRONTIER_AVAILABLE = True
except ImportError:
    _FRONTIER_AVAILABLE = False


@unittest.skipUnless(_FRONTIER_AVAILABLE, "Frontier not installed / not on PYTHONPATH")
class TestFrontierBatchDurationPredictorAdapt(unittest.TestCase):
    def setUp(self) -> None:
        ctx = build_monolithic_context(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=10.0,
        )
        self.fp = ctx.predictor
        self.wrap = FrontierBatchDurationPredictor(
            self.fp,
            cluster_type=ClusterType.MONOLITHIC,
            replica_id=0,
            is_moe=False,
        )

    def test_adapt_prefill_not_complete(self) -> None:
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=4,
            num_computed_tokens=0,
        )
        sb = ScheduleBatch(
            batch_id=1,
            requests=[req],
            tokens_per_request={1: 8},
            chunks=[PrefillChunk(request=req, num_tokens=8)],
        )
        fb = self.wrap._adapt(sb)
        self.assertEqual(len(fb.requests), 1)
        self.assertFalse(fb.requests[0].is_prefill_complete)
        self.assertEqual(fb.num_tokens[0], 8)
        self.assertEqual(fb.num_prefill_tokens, 8)

    def test_adapt_decode_marks_prefill_complete(self) -> None:
        req = InferenceRequest(
            request_id=2,
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=4,
            num_computed_tokens=16,
        )
        sb = ScheduleBatch(
            batch_id=2,
            requests=[req],
            tokens_per_request={2: 1},
            chunks=[DecodeChunk(request=req, num_tokens=1)],
        )
        fb = self.wrap._adapt(sb)
        self.assertTrue(fb.requests[0].is_prefill_complete)
        self.assertEqual(fb.num_tokens[0], 1)
        self.assertEqual(fb.num_prefill_tokens, 0)


@unittest.skipUnless(_FRONTIER_AVAILABLE, "Frontier not installed / not on PYTHONPATH")
class TestPredictAlignsWithFrontier(unittest.TestCase):
    """Wrapper duration must match direct Frontier predict on the adapted Batch."""

    def setUp(self) -> None:
        # RandomForrestExecutionTimePredictorConfig; bridge helper uses dummy for CI.
        # Alignment is wrap(sb) == predict_batch_duration(adapt(sb), same predictor).
        ctx = build_monolithic_context(
            replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1,
            dummy_execution_time_ms=10.0,
        )
        self.fp = ctx.predictor
        self.wrap = FrontierBatchDurationPredictor(
            self.fp,
            cluster_type=ClusterType.MONOLITHIC,
            is_moe=False,
        )
        self.gen = BatchLevelWorkloadGenerator(self.wrap)

    def _assert_aligned(self, sb: ScheduleBatch) -> None:
        got = self.wrap.predict(sb)
        adapted = self.wrap.last_adapted_batch
        self.assertIsNotNone(adapted)
        expect = predict_batch_duration(
            adapted, self.fp, cluster_type=ClusterType.MONOLITHIC
        )
        self.assertAlmostEqual(got, expect, places=9)

        wl = self.gen(sb, workload_id=42)
        self.assertEqual(wl["workload_id"], 42)
        self.assertEqual(len(wl["kernels"]), 1)
        self.assertAlmostEqual(wl["kernels"][0]["duration"], got, places=9)
        self.assertGreater(got, 0.0)

    def test_prefill_batch(self) -> None:
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=32,
            num_decode_tokens=2,
            num_computed_tokens=0,
        )
        sb = ScheduleBatch(
            batch_id=10,
            requests=[req],
            tokens_per_request={1: 16},
            chunks=[PrefillChunk(request=req, num_tokens=16)],
        )
        self._assert_aligned(sb)

    def test_decode_batch(self) -> None:
        req = InferenceRequest(
            request_id=2,
            arrived_at=0.0,
            num_prefill_tokens=32,
            num_decode_tokens=8,
            num_computed_tokens=32,
        )
        sb = ScheduleBatch(
            batch_id=11,
            requests=[req],
            tokens_per_request={2: 1},
            chunks=[DecodeChunk(request=req, num_tokens=1)],
        )
        self._assert_aligned(sb)

    def test_mixed_batch(self) -> None:
        prefill_req = InferenceRequest(
            request_id=3,
            arrived_at=0.0,
            num_prefill_tokens=24,
            num_decode_tokens=2,
            num_computed_tokens=0,
        )
        decode_req = InferenceRequest(
            request_id=4,
            arrived_at=0.0,
            num_prefill_tokens=24,
            num_decode_tokens=4,
            num_computed_tokens=24,
        )
        sb = ScheduleBatch(
            batch_id=12,
            requests=[prefill_req, decode_req],
            tokens_per_request={3: 8, 4: 1},
            chunks=[
                PrefillChunk(request=prefill_req, num_tokens=8),
                DecodeChunk(request=decode_req, num_tokens=1),
            ],
        )
        self._assert_aligned(sb)

    def test_factory_predict_mode(self) -> None:
        gen = make_infer_workload_generator(
            duration_mode="batch_level",
            batch_predictor="frontier",
            frontier_predictor=self.fp,
        )
        self.assertIsInstance(gen, BatchLevelWorkloadGenerator)
        req = InferenceRequest(
            request_id=5,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=1,
            num_computed_tokens=0,
        )
        sb = ScheduleBatch(
            batch_id=13,
            requests=[req],
            tokens_per_request={5: 8},
            chunks=[PrefillChunk(request=req, num_tokens=8)],
        )
        wl = gen(sb, workload_id=1)
        self.assertAlmostEqual(
            wl["kernels"][0]["duration"],
            self.wrap.predict(sb),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
