"""End-to-end: analytical duration_mode + model_preset through simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    build_inference_simulation,
)
from hybridsim_infer.workload_generators import (
    OpWorkloadGenerator,
    make_workload_generator,
)
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_analytical_config,
)


class TestSharedModelPreset(unittest.TestCase):
    def test_resolve_injects_preset_model(self) -> None:
        cfg = resolve_analytical_config(model_preset="llama-3.1-8b")
        self.assertIsNotNone(cfg)
        self.assertGreater(int(cfg.model.num_layers), 1)
        self.assertEqual(int(cfg.model.hidden_size), 4096)

    def test_factory_analytical_uses_preset(self) -> None:
        gen = make_workload_generator(
            duration_mode="analytical",
            model_preset="llama-3.1-8b",
        )
        self.assertIsInstance(gen, OpWorkloadGenerator)
        self.assertEqual(int(gen.config.model.hidden_size), 4096)


class TestAnalyticalE2E(unittest.TestCase):
    def test_analytical_preset_simulation_completes(self) -> None:
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            duration_mode="analytical",
            model_preset="llama-3.1-8b",
            tokens_per_step=8,
            decode_tokens_per_step=1,
            max_num_scheduled_tokens=64,
        )
        # Shrink layers for fast e2e while keeping preset shapes otherwise.
        analytical = resolve_analytical_config(model_preset="llama-3.1-8b")
        analytical.model.num_layers = 2
        cfg.analytical_config = analytical
        cfg.model_preset = None  # already injected

        infra = build_inference_simulation(cfg)
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=8,
            num_decode_tokens=2,
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)

    def test_analytical_with_prefix_caching(self) -> None:
        analytical = resolve_analytical_config(model_preset="llama-3.1-8b")
        analytical.model.num_layers = 2
        cfg = InferenceConfig(
            num_replicas=1,
            step_interval=1e-3,
            duration_mode="analytical",
            analytical_config=analytical,
            enable_prefix_caching=True,
            block_size=8,
            num_gpu_blocks=128,
            tokens_per_step=8,
            decode_tokens_per_step=1,
            max_num_scheduled_tokens=64,
        )
        infra = build_inference_simulation(cfg)
        # Seed local APC via manager if available.
        kv = infra.replicas[0]._kv
        prompt = list(range(16))
        if hasattr(kv, "cache_prefix"):
            kv.cache_prefix(prompt)

        r1 = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=1,
            prompt_token_ids=list(prompt),
        )
        r2 = InferenceRequest(
            request_id=2,
            arrived_at=0.05,
            num_prefill_tokens=16,
            num_decode_tokens=1,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([r1, r2])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 2)


if __name__ == "__main__":
    unittest.main()
