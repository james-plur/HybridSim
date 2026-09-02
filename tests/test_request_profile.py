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
from hybridsim_infer import (
    BatchFixedConfig,
    BatchLevelConfig,
    BatchTokenProportionalConfig,
    ClusterConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    KvConfig,
    KvLookupConfig,
    KvWorkloadConfig,
    ModelSpec,
    OutputConfig,
    ParallelConfig,
    ReplicaScheduleConfig,
    RequestProfileOutput,
    ScheduleConfig,
    build_inference_simulation,
)
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_op_level_config,
)


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
                cluster=ClusterConfig(num_replicas=1),
                schedule=ScheduleConfig(
                    replica=ReplicaScheduleConfig(
                        tokens_per_step=8,
                        max_num_scheduled_tokens=64,
                    ),
                ),
                infer_workload=InferWorkloadConfig(
                    batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.02)),
                ),
                output=OutputConfig(
                    request_profile=RequestProfileOutput(enabled=True, path=out),
                ),
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
                cluster=ClusterConfig(
                    type="pd",
                    num_prefill_replicas=1,
                    num_decode_replicas=1,
                ),
                schedule=ScheduleConfig(
                    replica=ReplicaScheduleConfig(
                        tokens_per_step=8,
                        decode_tokens_per_step=1,
                        max_num_scheduled_tokens=64,
                    ),
                ),
                kv=KvConfig(
                    enable_store=True,
                    enable_prefix_caching=False,
                    block_size=8,
                    num_gpu_blocks=256,
                    lookup=KvLookupConfig(rtt_s=1e-3),
                ),
                model=ModelSpec(preset="llama-3.1-8b"),
                infer_workload=InferWorkloadConfig(
                    mode="batch_level",
                    batch=BatchLevelConfig(
                        predictor="token_proportional",
                        fixed=BatchFixedConfig(dummy_exec_s=0.01),
                        token_proportional=BatchTokenProportionalConfig(
                            prefill_s_per_token=5e-5,
                            decode_s_per_token=2e-4,
                        ),
                    ),
                ),
                kv_workload=KvWorkloadConfig(
                    bandwidth_gbps=100.0,
                    transfer_s_floor=1e-4,
                ),
                output=OutputConfig(
                    request_profile=RequestProfileOutput(enabled=True, path=out),
                ),
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
            self.assertGreaterEqual(len(_events_by_name(profile, "Handoff")), 1)
            self.assertGreaterEqual(len(_events_by_name(profile, "RequestFinish")), 1)
            proc_names = {
                e["args"]["name"]
                for e in profile["traceEvents"]
                if e.get("name") == "process_name"
            }
            self.assertTrue(any("Prefill" in n for n in proc_names))
            self.assertTrue(any("Decode" in n for n in proc_names))


class TestRequestProfileOpLevel(unittest.TestCase):
    def test_kernel_streams_and_critical_path_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "op.json"
            op_level = resolve_op_level_config(model_preset="llama-3.1-8b")
            assert op_level is not None
            op_level.model.num_layers = 2
            op_level.parallel = ParallelConfig(tp_size=2)
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
                output=OutputConfig(
                    request_profile=RequestProfileOutput(enabled=True, path=out),
                ),
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
            profile = _load_profile(out)
            engines = _events_by_name(profile, "EngineReq")
            self.assertGreaterEqual(len(engines), 1)
            for eng in engines:
                args = eng["args"]
                self.assertIn("phase", args)
                self.assertIn("n_kernels", args)
                self.assertIn("critical_path_s", args)
                self.assertGreater(int(args["n_kernels"]), 1)
                self.assertAlmostEqual(
                    float(eng["dur"]) / 1e6,
                    float(args["critical_path_s"]),
                    places=6,
                )
                self.assertGreater(float(eng["dur"]), 0.0)

            thread_names = {
                e["args"]["name"]
                for e in profile["traceEvents"]
                if e.get("name") == "thread_name"
            }
            self.assertTrue(any(n.startswith("compute_") for n in thread_names))
            self.assertTrue(any(n.startswith("comm_") for n in thread_names))
            kernel_events = [
                e
                for e in profile["traceEvents"]
                if e.get("ph") == "X" and e.get("cat") == "op_kernel"
            ]
            self.assertGreater(len(kernel_events), 0)
            self.assertTrue(any("gemm" in str(e.get("name", "")).lower() or "attn" in str(e.get("name", "")).lower() for e in kernel_events))
            kernel_deps = [
                e for e in profile["traceEvents"] if e.get("name") == "KernelDep"
            ]
            self.assertGreater(len(kernel_deps), 0)


class TestCreateSessionDisabled(unittest.TestCase):
    def test_null_session(self) -> None:
        session = create_request_profile_session(enabled=False)
        self.assertFalse(session.enabled)
        self.assertIsNone(session.stop())


if __name__ == "__main__":
    unittest.main()
