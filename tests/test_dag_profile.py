"""Tests for analytical Op DAG Chrome Trace profile export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.workload_generators.analytic_model.configs import (
    AnalyticalConfig,
    ModelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.analytic_model.dag_profile import (
    asap_schedule,
    build_chrome_trace,
    make_demo_prefill_batch,
    profile_schedule_batch,
    rank_layout,
    summarize_overlap,
    write_chrome_trace,
)
from hybridsim_infer.workload_generators.analytic_model.types import AttnVariant


class TestDagProfile(unittest.TestCase):
    def test_asap_respects_deps(self) -> None:
        kernels = [
            {"name": "a", "duration": 1.0, "dependencies": []},
            {"name": "b", "duration": 2.0, "dependencies": [0]},
            {"name": "c", "duration": 1.0, "dependencies": [0]},
            {"name": "d", "duration": 1.0, "dependencies": [1, 2]},
        ]
        sched = asap_schedule(kernels)
        self.assertEqual(sched[0].start_s, 0.0)
        self.assertEqual(sched[1].start_s, 1.0)
        self.assertEqual(sched[2].start_s, 1.0)  # overlap with b
        self.assertEqual(sched[3].start_s, 3.0)  # after longer of b/c
        stats = summarize_overlap(sched)
        self.assertAlmostEqual(stats["critical_path_s"], 4.0)
        self.assertGreater(stats["overlap_ratio"], 0.0)

    def test_rank_layout_tp_pp(self) -> None:
        ranks = rank_layout(ParallelConfig(tp_size=2, pp_size=2))
        self.assertEqual(len(ranks), 4)
        self.assertEqual(ranks[0], (0, 0, 0))
        self.assertEqual(ranks[-1], (3, 1, 1))

    def test_chrome_trace_structure(self) -> None:
        cfg = AnalyticalConfig(
            model=ModelConfig(
                num_layers=2,
                hidden_size=1024,
                intermediate_size=2816,
                num_q_heads=16,
                num_kv_heads=16,
                head_dim=64,
                attn_variant=AttnVariant.MHA,
            ),
            parallel=ParallelConfig(tp_size=2, pp_size=1),
        )
        batch = make_demo_prefill_batch(chunk=32, cached=16)
        trace = profile_schedule_batch(batch, analytical=cfg)
        events = trace["traceEvents"]
        self.assertTrue(any(e.get("ph") == "M" for e in events))
        completes = [e for e in events if e.get("ph") == "X"]
        self.assertGreater(len(completes), 0)
        pids = {e["pid"] for e in completes}
        self.assertEqual(len(pids), 2)  # tp=2
        self.assertIn("critical_path_s", trace["otherData"])

        with tempfile.TemporaryDirectory() as tmp:
            path = write_chrome_trace(trace, Path(tmp) / "t.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["traceEvents"]), len(events))

    def test_preset_resolve_in_profile(self) -> None:
        batch = make_demo_prefill_batch(chunk=16, cached=0)
        trace = profile_schedule_batch(
            batch,
            analytical=AnalyticalConfig(parallel=ParallelConfig(tp_size=1)),
            model_preset="llama-3.1-8b",
        )
        self.assertGreater(trace["otherData"]["num_kernels"], 0)


if __name__ == "__main__":
    unittest.main()
