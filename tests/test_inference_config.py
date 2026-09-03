"""Tests for nested InferenceConfig validation and optional artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hybridsim_infer import (
    ArtifactOutput,
    ClusterConfig,
    InferenceConfig,
    InferenceRequest,
    OutputConfig,
    build_inference_simulation,
    format_metrics,
)


class TestInferenceConfigValidate(unittest.TestCase):
    def test_invalid_cluster_type_raises(self) -> None:
        cfg = InferenceConfig(cluster=ClusterConfig(type="mesh"))
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("monolith", str(ctx.exception))

    def test_pd_requires_both_pools(self) -> None:
        cfg = InferenceConfig(
            cluster=ClusterConfig(type="pd", num_prefill_replicas=0, num_decode_replicas=1)
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_store_block_must_be_multiple(self) -> None:
        from hybridsim_infer import KvConfig, KvStoreConfig

        cfg = InferenceConfig(
            kv=KvConfig(block_size=16, store=KvStoreConfig(block_size=24))
        )
        with self.assertRaises(ValueError):
            cfg.validate()


class TestInferenceOutputs(unittest.TestCase):
    def test_metrics_and_requests_written_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cfg = InferenceConfig(
                output=OutputConfig(
                    dir=out_dir,
                    metrics=ArtifactOutput(enabled=True),
                    requests=ArtifactOutput(enabled=True),
                    config_snapshot=ArtifactOutput(enabled=True),
                )
            )
            infra = build_inference_simulation(cfg)
            infra.schedule_arrivals(
                [
                    InferenceRequest(
                        request_id=1,
                        arrived_at=0.0,
                        num_prefill_tokens=8,
                        num_decode_tokens=2,
                    )
                ]
            )
            infra.run()
            infra.check_errors()
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["n_finished"], 1)
            self.assertEqual(metrics["n_scheduled"], 1)
            rows = (out_dir / "requests.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            rec = json.loads(rows[0])
            self.assertEqual(rec["request_id"], 1)
            snap = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(snap["cluster"]["type"], "monolith")
            self.assertIn("op", snap["infer_workload"])
            self.assertIn("model", snap["infer_workload"]["op"])


class TestFormatMetrics(unittest.TestCase):
    def test_aligned_human_readable_block(self) -> None:
        text = format_metrics(
            {
                "hit_rate": 0.1467208864386889,
                "mean_ttft_s": 15.494274247807946,
                "n_finished": 10,
                "n_scheduled": 10,
                "prefill_tokens": 73282,
                "prefix_hit_tokens": 10752,
                "sim_now_s": 34.14999967296004,
                "tps": 2145.885818500451,
            }
        )
        self.assertIn("10 finished / 10 scheduled", text)
        self.assertIn("34.150 s", text)
        self.assertIn("15.494 s", text)
        self.assertIn("2,145.9 tok/s", text)
        self.assertIn("14.67%", text)
        self.assertIn("10,752 / 73,282", text)

    def test_empty_run(self) -> None:
        text = format_metrics(
            {
                "mean_ttft_s": None,
                "tps": 0.0,
                "hit_rate": 0.0,
                "n_finished": 0,
                "n_scheduled": 0,
                "sim_now_s": 0.0,
                "prefill_tokens": 0,
                "prefix_hit_tokens": 0,
            }
        )
        self.assertIn("n/a", text)
        self.assertIn("0.00%", text)


class TestInferWorkloadOp(unittest.TestCase):
    def test_op_defaults_to_op_level_config(self) -> None:
        from hybridsim_infer import InferWorkloadConfig, OpLevelConfig

        cfg = InferenceConfig()
        self.assertIsInstance(cfg.infer_workload.op, OpLevelConfig)
        self.assertIsInstance(InferWorkloadConfig().op, OpLevelConfig)


if __name__ == "__main__":
    unittest.main()
