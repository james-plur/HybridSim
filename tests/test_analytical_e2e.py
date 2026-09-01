"""End-to-end: op_level duration_mode + model_preset through simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer import (
    ClusterConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    KvConfig,
    ReplicaScheduleConfig,
    ScheduleConfig,
    build_inference_simulation,
)
from hybridsim_infer.workload_generators import (
    OpLevelWorkloadGenerator,
    make_infer_workload_generator,
)
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_op_level_config,
)


class TestSharedModelPreset(unittest.TestCase):
    def test_resolve_injects_preset_model(self) -> None:
        cfg = resolve_op_level_config(model_preset="llama-3.1-8b")
        self.assertIsNotNone(cfg)
        self.assertGreater(int(cfg.model.num_layers), 1)
        self.assertEqual(int(cfg.model.hidden_size), 4096)

    def test_factory_op_level_uses_preset(self) -> None:
        gen = make_infer_workload_generator(
            duration_mode="op_level",
            model_preset="llama-3.1-8b",
        )
        self.assertIsInstance(gen, OpLevelWorkloadGenerator)
        self.assertEqual(int(gen.config.model.hidden_size), 4096)


class TestOpLevelE2E(unittest.TestCase):
    def test_op_level_preset_simulation_completes(self) -> None:
        op_level = resolve_op_level_config(model_preset="llama-3.1-8b")
        op_level.model.num_layers = 2
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                    max_num_scheduled_tokens=64,
                ),
            ),
            infer_workload=InferWorkloadConfig(mode="op_level", op=op_level),
        )

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

    def test_op_level_with_prefix_caching(self) -> None:
        op_level = resolve_op_level_config(model_preset="llama-3.1-8b")
        op_level.model.num_layers = 2
        cfg = InferenceConfig(
            cluster=ClusterConfig(num_replicas=1),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(
                    tokens_per_step=8,
                    decode_tokens_per_step=1,
                    max_num_scheduled_tokens=64,
                ),
            ),
            kv=KvConfig(
                enable_prefix_caching=True,
                block_size=8,
                num_gpu_blocks=128,
            ),
            infer_workload=InferWorkloadConfig(mode="op_level", op=op_level),
        )
        infra = build_inference_simulation(cfg)
        kv = infra.replicas[0]._kv
        prompt = list(range(16))
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
        by_id = {r.request_id: r for r in infra.finished_requests}
        self.assertIsNotNone(by_id[1].finished_at)
        self.assertGreaterEqual(float(by_id[1].finished_at), 0.0)
        self.assertGreater(by_id[2].prefix_hit_tokens, 0)


if __name__ == "__main__":
    unittest.main()
