"""Unit tests for Mooncake schedule + Store pool profile alignment."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

# Ensure reproducible hashes before importing hybridsim hash helpers.
os.environ.setdefault("PYTHONHASHSEED", "0")

from hybridsim_infer import (
    BatchFixedConfig,
    BatchLevelConfig,
    ClusterConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    KvConfig,
    ReplicaScheduleConfig,
    ScheduleConfig,
    build_inference_simulation,
)
from hybridsim_infer.kv_system import block_keys_from_tokens, reset_none_hash

from mooncake_alignment.compare import compare_pool_profiles
from mooncake_alignment.hybridsim_store_driver import run_hybridsim_store_case
from mooncake_alignment.schema import MooncakePoolEvent, write_pool_profile
from schedule_alignment.case_loader import CaseSpec
from schedule_alignment.compare import compare_ledgers
from schedule_alignment.schema import write_ledger

_ROOT = Path(__file__).resolve().parent
_CASES = _ROOT / "mooncake_alignment" / "cases"
_OUT = _ROOT / "mooncake_alignment" / "_out"


def _load_case(stem: str) -> CaseSpec:
    path = _CASES / f"{stem}.json"
    with path.open(encoding="utf-8") as fh:
        return CaseSpec.from_dict(json.load(fh), name=stem)


def _assert_deterministic(case: CaseSpec):
    sched, pool = run_hybridsim_store_case(case)
    sched2, pool2 = run_hybridsim_store_case(case)
    assert compare_ledgers(sched, sched2).equal
    assert compare_pool_profiles(pool, pool2).ok
    return sched, pool


class TestStorePrefixHit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reset_none_hash()
        _OUT.mkdir(parents=True, exist_ok=True)

    def test_hybridsim_schedule_and_pool_self_consistent(self) -> None:
        case = _load_case("store_prefix_hit")
        sched, pool = _assert_deterministic(case)
        self.assertTrue(any(e.op == "put" for e in pool))
        exists = [e for e in pool if e.op == "exist" and e.num_tokens > 0]
        self.assertTrue(exists)
        write_ledger(_OUT / "store_prefix_hit.hybridsim.ledger.jsonl", sched)
        write_pool_profile(_OUT / "store_prefix_hit.hybridsim.pool.jsonl", pool)

    def test_block_keys_stable(self) -> None:
        reset_none_hash()
        a = block_keys_from_tokens(list(range(8)), 8)
        b = block_keys_from_tokens(list(range(8)), 8)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(a[0]), 64)  # sha256 hex


class TestStorePrefixPartialHit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reset_none_hash()
        _OUT.mkdir(parents=True, exist_ok=True)

    def test_partial_remote_hit_then_suffix_prefill(self) -> None:
        case = _load_case("store_prefix_partial_hit")
        sched, pool = _assert_deterministic(case)

        # Req2 longest contiguous exist is exactly 2 blocks (16 tokens).
        partial = [
            e
            for e in pool
            if e.op == "exist" and e.req_id == "2" and e.num_tokens == 16
        ]
        self.assertTrue(partial, pool)
        self.assertEqual(partial[0].hit_mask, [True, True, False])

        # After remote get, remaining prefill for req2 is one block (8 tokens).
        req2_sched = [
            rec.scheduled_tokens.get("2", 0)
            for rec in sched
            if "2" in rec.scheduled_tokens
        ]
        self.assertEqual(sum(req2_sched), 8, req2_sched)

        # Divergent suffix is put (1 block); shared prefix not re-inserted.
        puts_r2 = [e for e in pool if e.op == "put" and e.req_id == "2"]
        self.assertEqual(sum(e.num_tokens for e in puts_r2), 8)

        write_ledger(_OUT / "store_prefix_partial_hit.hybridsim.ledger.jsonl", sched)
        write_pool_profile(
            _OUT / "store_prefix_partial_hit.hybridsim.pool.jsonl", pool
        )

    def test_nested_partial_hits(self) -> None:
        case = _load_case("store_prefix_nested_partial")
        _, pool = _assert_deterministic(case)

        hits_r2 = max(
            (e.num_tokens for e in pool if e.op == "exist" and e.req_id == "2"),
            default=0,
        )
        hits_r3 = max(
            (e.num_tokens for e in pool if e.op == "exist" and e.req_id == "3"),
            default=0,
        )
        self.assertEqual(hits_r2, 16)  # shares 2 blocks with req1
        self.assertEqual(hits_r3, 8)  # shares only first block


class TestStorePrefixEvict(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reset_none_hash()
        _OUT.mkdir(parents=True, exist_ok=True)

    def test_lru_evict_then_miss(self) -> None:
        case = _load_case("store_prefix_evict")
        self.assertEqual(case.scheduler.get("store_num_blocks"), 2)
        sched, pool = _assert_deterministic(case)

        evicts = [e for e in pool if e.op == "evict"]
        self.assertGreaterEqual(len(evicts), 2, pool)

        # Req2 put must have triggered eviction of Req1's blocks.
        evict_on_r2 = [e for e in evicts if e.req_id == "2"]
        self.assertTrue(evict_on_r2)

        # Req3 reuses Req1 prompt but store capacity flushed those keys → miss.
        exists_r3 = [e for e in pool if e.op == "exist" and e.req_id == "3"]
        self.assertTrue(exists_r3)
        self.assertTrue(all(e.num_tokens == 0 for e in exists_r3), exists_r3)

        # Full prefill again for req3 (16 tokens across steps).
        req3_sched = sum(
            rec.scheduled_tokens.get("3", 0)
            for rec in sched
            if "3" in rec.scheduled_tokens
        )
        self.assertEqual(req3_sched, 16)

        write_ledger(_OUT / "store_prefix_evict.hybridsim.ledger.jsonl", sched)
        write_pool_profile(_OUT / "store_prefix_evict.hybridsim.pool.jsonl", pool)


class TestPdHandoffDes(unittest.TestCase):
    def test_pd_des_completes(self) -> None:
        prompt = list(range(20, 28))
        cfg = InferenceConfig(
            cluster=ClusterConfig(
                type="pd",
                num_prefill_replicas=1,
                num_decode_replicas=1,
            ),
            schedule=ScheduleConfig(
                replica=ReplicaScheduleConfig(tokens_per_step=8),
            ),
            kv=KvConfig(enable_store=True, block_size=8),
            infer_workload=InferWorkloadConfig(
                batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
            ),
        )
        infra = build_inference_simulation(cfg)
        req = InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt),
            num_decode_tokens=2,
            prompt_token_ids=list(prompt),
        )
        infra.schedule_arrivals([req])
        infra.run()
        infra.check_errors()
        self.assertEqual(len(infra.finished_requests), 1)
        self.assertTrue(infra.finished_requests[0].completed)


class TestEnvScriptsPresent(unittest.TestCase):
    def test_scripts_exist(self) -> None:
        scripts = _ROOT / "mooncake_alignment" / "scripts"
        self.assertTrue((scripts / "check_env.sh").exists())
        self.assertTrue((scripts / "start_master.sh").exists())
        self.assertTrue((scripts / "mooncake_config.tcp.json").exists())


class TestPoolCompare(unittest.TestCase):
    def test_compare_detects_mismatch(self) -> None:
        a = [MooncakePoolEvent(step=0, op="put", hashes=["aa"], num_tokens=8)]
        b = [MooncakePoolEvent(step=0, op="put", hashes=["bb"], num_tokens=8)]
        report = compare_pool_profiles(a, b)
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
