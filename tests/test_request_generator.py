"""Tests for RequestGenerator + ServeGen mapping / optional integration."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    ListRequestGenerator,
    build_inference_simulation,
    map_servegen_request,
)
from hybridsim_infer.request_generators.servegen_generator import (
    ServeGenRequestGenerator,
)

try:
    import servegen  # noqa: F401

    _HAS_SERVEGEN = True
except ImportError:
    _HAS_SERVEGEN = False


@dataclass
class _FakeServeGenRequest:
    request_id: int
    timestamp: float
    data: dict[str, Any]


class TestListRequestGenerator(unittest.TestCase):
    def test_generate_returns_copy(self) -> None:
        reqs = [
            InferenceRequest(request_id=1, arrived_at=0.0, num_prefill_tokens=4, num_decode_tokens=2),
            InferenceRequest(request_id=2, arrived_at=0.1, num_prefill_tokens=8, num_decode_tokens=1),
        ]
        gen = ListRequestGenerator(reqs)
        out = gen.generate()
        self.assertEqual(len(out), 2)
        self.assertIsNot(out, reqs)
        self.assertEqual(out[0].request_id, 1)

    def test_schedule_from_generator(self) -> None:
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            dummy_exec_s=0.01,
            tokens_per_step=8,
            max_num_scheduled_tokens=64,
        )
        infra = build_inference_simulation(cfg)
        reqs = [
            InferenceRequest(
                request_id=10,
                arrived_at=0.0,
                num_prefill_tokens=4,
                num_decode_tokens=2,
            ),
            InferenceRequest(
                request_id=11,
                arrived_at=0.05,
                num_prefill_tokens=4,
                num_decode_tokens=2,
            ),
        ]
        scheduled = infra.schedule_from_generator(ListRequestGenerator(reqs))
        self.assertEqual(len(scheduled), 2)
        infra.run()
        infra.check_errors()
        self.assertEqual(infra.cluster.arrived_count, 2)
        self.assertEqual(len(infra.finished_requests), 2)


class TestMapServeGenRequest(unittest.TestCase):
    def test_language_mapping(self) -> None:
        raw = _FakeServeGenRequest(
            request_id=3,
            timestamp=1.25,
            data={"input_tokens": 128, "output_tokens": 16},
        )
        req = map_servegen_request(raw, id_offset=100, time_offset=10.0)
        self.assertEqual(req.request_id, 103)
        self.assertAlmostEqual(req.arrived_at, 11.25)
        self.assertEqual(req.num_prefill_tokens, 128)
        self.assertEqual(req.num_decode_tokens, 16)

    def test_missing_fields_raises(self) -> None:
        raw = _FakeServeGenRequest(
            request_id=0,
            timestamp=0.0,
            data={"image_tokens": [1, 2]},
        )
        with self.assertRaises(ValueError):
            map_servegen_request(raw)


class TestServeGenOptionalDependency(unittest.TestCase):
    @unittest.skipIf(_HAS_SERVEGEN, "servegen installed")
    def test_construct_raises_without_servegen(self) -> None:
        with self.assertRaises(ImportError):
            ServeGenRequestGenerator()


@unittest.skipUnless(_HAS_SERVEGEN, "servegen not installed")
class TestServeGenRequestGenerator(unittest.TestCase):
    def test_generate_language_requests(self) -> None:
        gen = ServeGenRequestGenerator(
            model="m-small",
            duration=60,
            rate=2.0,
            seed=0,
            max_requests=5,
        )
        reqs = gen.generate()
        self.assertGreater(len(reqs), 0)
        self.assertLessEqual(len(reqs), 5)
        prev_t = -1.0
        for r in reqs:
            self.assertGreaterEqual(r.arrived_at, prev_t)
            prev_t = r.arrived_at
            self.assertGreaterEqual(r.num_prefill_tokens, 0)
            self.assertGreaterEqual(r.num_decode_tokens, 0)

    def test_short_simulation(self) -> None:
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            duration_mode="batch_level",
            batch_predictor="token_proportional",
            prefill_s_per_token=1e-5,
            decode_s_per_token=1e-4,
            tokens_per_step=64,
            max_num_scheduled_tokens=256,
            max_num_running_reqs=16,
            num_gpu_blocks=4096,
        )
        infra = build_inference_simulation(cfg)
        gen = ServeGenRequestGenerator(
            model="m-small",
            duration=60,
            rate=2.0,
            seed=0,
            max_requests=3,
        )
        reqs = infra.schedule_from_generator(gen)
        infra.run()
        infra.check_errors()
        self.assertEqual(infra.cluster.arrived_count, len(reqs))
        self.assertEqual(len(infra.finished_requests), len(reqs))

    def test_non_language_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ServeGenRequestGenerator(category="multimodal")


if __name__ == "__main__":
    unittest.main()
