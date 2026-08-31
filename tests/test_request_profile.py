"""Tests for request-level Chrome Trace profiling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hybridsim.request_profile import (
    RequestProfileSession,
    create_request_profile_session,
)
from hybridsim_infer import InferenceConfig, InferenceRequest, build_inference_simulation


def _load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _events_by_name(profile: dict, name: str) -> list[dict]:
    return [e for e in profile.get("traceEvents", []) if e.get("name") == name]


class TestRequestProfileSession(unittest.TestCase):
    def test_process_cleanup_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "req.json"
            session = RequestProfileSession(out)
            session.start()
            self.assertTrue(session.process_alive)
            session.emit_cluster_schedule(time_s=0.0)
            session.emit_dispatch(
                time_s=0.0, request_id=1, replica_id=0, kind="arrive"
            )
            path = session.stop(timeout=10.0)
            self.assertFalse(session.process_alive)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            profile = _load_profile(path)
            self.assertIn("traceEvents", profile)
            self.assertEqual(profile.get("displayTimeUnit"), "ms")


class TestRequestProfileMonolith(unittest.TestCase):
    def test_format_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mono.json"
            cfg = InferenceConfig(
                num_replicas=1,
                step_interval=1e-3,
                dummy_exec_s=0.02,
                tokens_per_step=8,
                max_num_scheduled_tokens=64,
                enable_request_profile=True,
                request_profile_path=out,
            )
            infra = build_inference_simulation(cfg)
            infra.schedule_arrivals(
                [
                    InferenceRequest(
                        request_id=1,
                        arrived_at=0.0,
                        num_prefill_tokens=8,
                        num_decode_tokens=2,
                    ),
                    InferenceRequest(
                        request_id=2,
                        arrived_at=0.01,
                        num_prefill_tokens=4,
                        num_decode_tokens=2,
                    ),
                ]
            )
            infra.run()
            infra.check_errors()
            self.assertTrue(out.exists())
            profile = _load_profile(out)
            self.assertIn("traceEvents", profile)
            self.assertEqual(profile["displayTimeUnit"], "ms")

            names = {e.get("name") for e in profile["traceEvents"]}
            self.assertIn("process_name", names)
            self.assertIn("thread_name", names)
            self.assertIn("Dispatch", names)
            self.assertIn("EngineReq", names)
            self.assertIn("ReplicaSchedule", names)
            self.assertIn("ClusterSchedule", names)
            self.assertIn("ReplicaEnqueue", names)
            self.assertIn("ClusterToReplica", names)
            self.assertIn("ScheduleToEngine", names)

            for e in profile["traceEvents"]:
                if e.get("ph") == "X":
                    self.assertGreaterEqual(float(e.get("dur", 0)), 0.0)

            # Flow start/end pairs share id.
            flows = [
                e
                for e in profile["traceEvents"]
                if e.get("ph") in ("s", "f")
            ]
            by_id: dict[int, dict[str, list]] = {}
            for e in flows:
                by_id.setdefault(int(e["id"]), {"s": [], "f": []})[e["ph"]].append(e)
            self.assertGreaterEqual(len(by_id), 2)
            for fid, pair in by_id.items():
                self.assertEqual(len(pair["s"]), 1, msg=f"flow {fid} start")
                self.assertEqual(len(pair["f"]), 1, msg=f"flow {fid} end")
                self.assertGreaterEqual(
                    float(pair["f"][0]["ts"]), float(pair["s"][0]["ts"])
                )

            dispatches = {
                int(e["args"]["request_id"]): e for e in _events_by_name(profile, "Dispatch")
            }
            engines = _events_by_name(profile, "EngineReq")
            self.assertGreaterEqual(len(dispatches), 2)
            self.assertGreaterEqual(len(engines), 1)
            for eng in engines:
                rid = int(eng["args"]["request_id"])
                self.assertIn(rid, dispatches)
                self.assertGreaterEqual(
                    float(eng["ts"]), float(dispatches[rid]["ts"])
                )
                # Replica_0 → pid 2
                self.assertEqual(int(eng["pid"]), 2)

            requests_meta = profile.get("metadata", {}).get("requests", {})
            self.assertIn("1", requests_meta)
            self.assertEqual(requests_meta["1"]["num_prefill_tokens"], 8)
            self.assertEqual(requests_meta["1"]["num_decode_tokens"], 2)
            self.assertIn("arrived_at", requests_meta["1"])
            self.assertTrue(requests_meta["1"].get("completed"))
            self.assertIn("finished_at", requests_meta["1"])
            # Dispatch args carry a compact meta subset for UI hover.
            self.assertEqual(dispatches[1]["args"]["num_prefill_tokens"], 8)


class TestRequestProfilePdKv(unittest.TestCase):
    def test_handoff_and_kv_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pd.json"
            cfg = InferenceConfig(
                cluster_type="pd",
                num_prefill_replicas=1,
                num_decode_replicas=1,
                enable_kv_client=True,
                enable_prefix_caching=False,
                model_preset="llama-3.1-8b",
                block_size=8,
                num_gpu_blocks=256,
                tokens_per_step=8,
                decode_tokens_per_step=1,
                max_num_scheduled_tokens=64,
                step_interval=1e-3,
                dummy_exec_s=0.01,
                kv_transfer_s=1e-4,
                kv_bandwidth_gbps=100.0,
                kv_lookup_rtt_s=1e-3,
                duration_mode="batch_level",
                batch_predictor="token_proportional",
                prefill_s_per_token=5e-5,
                decode_s_per_token=2e-4,
                enable_request_profile=True,
                request_profile_path=out,
            )
            infra = build_inference_simulation(cfg)
            prompt = list(range(16))
            infra.schedule_arrivals(
                [
                    InferenceRequest(
                        request_id=1,
                        arrived_at=0.0,
                        num_prefill_tokens=len(prompt),
                        num_decode_tokens=4,
                        prompt_token_ids=list(prompt),
                    )
                ]
            )
            infra.run()
            infra.check_errors()
            self.assertEqual(len(infra.finished_requests), 1)
            profile = _load_profile(out)
            dispatches = _events_by_name(profile, "Dispatch")
            kinds = {e["args"].get("kind") for e in dispatches}
            self.assertIn("arrive", kinds)
            self.assertIn("handoff", kinds)
            kv_pulls = _events_by_name(profile, "KvPull")
            self.assertGreaterEqual(len(kv_pulls), 1)


class TestCreateSessionDisabled(unittest.TestCase):
    def test_null_session(self) -> None:
        session = create_request_profile_session(enabled=False)
        self.assertFalse(session.enabled)
        self.assertIsNone(session.stop())


if __name__ == "__main__":
    unittest.main()
