"""Unit tests for KV / Mooncake management alignment (save gate, APC, SSD)."""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any
from unittest.mock import MagicMock

os.environ.setdefault("PYTHONHASHSEED", "0")

from hybridsim_infer.kv_system import MooncakeKvStore, VllmKvCacheManager, block_keys_from_tokens
from hybridsim_infer.kv_system.block_keys import reset_none_hash
from hybridsim_infer.kv_system.client import KvClient
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.workload_generators.kv_transfer import transfer_duration_s


class SaveGateIncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_none_hash()

    def test_gate_skips_until_full_block(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=64, block_size=16, enable_prefix_caching=False
        )
        pushes: list[int] = []

        class _Client:
            has_store = True

            async def save(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                keys = kwargs.get("block_keys") or []
                return {
                    "ok": True,
                    "num_tokens": len(keys) * 16,
                    "num_blocks": len(keys),
                    "cached": False,
                }

            def submit_push(self, rid: int, n: int) -> None:
                pushes.append(n)

        kv.attach_client(_Client())  # type: ignore[arg-type]
        req = InferenceRequest(
            request_id=1,
            num_prefill_tokens=40,
            num_decode_tokens=1,
            prompt_token_ids=list(range(40)),
            status=RequestStatus.RUNNING,
            num_computed_tokens=8,
        )
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(pushes, [])
        self.assertEqual(kv._num_saved_tokens.get(1, 0), 0)

        req.num_computed_tokens = 16
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(pushes, [16])
        self.assertEqual(kv._num_saved_tokens[1], 16)

        req.num_computed_tokens = 32
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(pushes, [16, 16])
        self.assertEqual(kv._num_saved_tokens[1], 32)

    def test_confirm_cached_skips_push(self) -> None:
        kv = VllmKvCacheManager(num_gpu_blocks=64, block_size=16)
        pushes: list[int] = []

        class _Client:
            has_store = True

            async def save(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"ok": True, "num_tokens": 0, "num_blocks": 0, "cached": True}

            def submit_push(self, rid: int, n: int) -> None:
                pushes.append(n)

        kv.attach_client(_Client())  # type: ignore[arg-type]
        req = InferenceRequest(
            request_id=7,
            num_prefill_tokens=16,
            num_decode_tokens=1,
            prompt_token_ids=list(range(16)),
            status=RequestStatus.RUNNING,
            num_computed_tokens=16,
        )
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(pushes, [])
        self.assertEqual(kv._num_saved_tokens[7], 16)


class BlockPoolApcTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_none_hash()

    def test_reuse_does_not_double_count_free(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=16, block_size=16, enable_prefix_caching=True
        )
        usable = kv.free_blocks
        r1 = InferenceRequest(
            request_id=1,
            num_prefill_tokens=32,
            num_decode_tokens=1,
            prompt_token_ids=list(range(32)),
        )
        self.assertIsNotNone(kv.allocate(r1, 32))
        self.assertEqual(kv.free_blocks, usable - 2)
        kv.free(r1)
        self.assertEqual(kv.free_blocks, usable)

        r2 = InferenceRequest(
            request_id=2,
            num_prefill_tokens=32,
            num_decode_tokens=1,
            prompt_token_ids=list(range(32)),
        )
        hit = kv.match(r2)
        self.assertEqual(hit, 16)
        attached = kv.attach_cached_prefix(r2, hit)
        self.assertIsNotNone(attached)
        self.assertEqual(len(attached or []), 1)
        # One reused block pulled from free queue.
        self.assertEqual(kv.free_blocks, usable - 1)
        kv.free(r2)
        self.assertEqual(kv.free_blocks, usable)

    def test_mid_flight_visibility(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=32, block_size=16, enable_prefix_caching=True
        )
        a = InferenceRequest(
            request_id=10,
            num_prefill_tokens=48,
            num_decode_tokens=1,
            prompt_token_ids=list(range(48)),
            status=RequestStatus.RUNNING,
        )
        self.assertIsNotNone(kv.allocate(a, 32))
        a.num_computed_tokens = 32
        # A still running; B should see A's full blocks.
        b = InferenceRequest(
            request_id=11,
            num_prefill_tokens=48,
            num_decode_tokens=1,
            prompt_token_ids=list(range(48)),
        )
        self.assertEqual(kv.match(b), 32)
        attached = kv.attach_cached_prefix(b, 32)
        self.assertEqual(len(attached or []), 2)
        blk = kv.allocated[10][0]
        self.assertGreaterEqual(blk.ref_cnt, 2)

    def test_unlimited_gpu_blocks(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=0, block_size=16, enable_prefix_caching=True
        )
        r = InferenceRequest(
            request_id=1,
            num_prefill_tokens=64,
            num_decode_tokens=1,
            prompt_token_ids=list(range(64)),
        )
        self.assertIsNotNone(kv.allocate(r, 64))
        self.assertEqual(len(kv.allocated[1]), 4)


class StoreCapacitySsdTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_none_hash()

    def test_unlimited_dram_no_evict(self) -> None:
        store = MooncakeKvStore(num_blocks=0, block_size=16)
        keys = [f"k{i}" for i in range(100)]
        put = store.insert_keys(keys, req_id="1")
        self.assertTrue(put["ok"])
        self.assertEqual(put["num_tokens"], 100 * 16)
        self.assertEqual(len(store.snapshot_hashes()), 100)

    def test_insert_returns_incremental_only(self) -> None:
        store = MooncakeKvStore(num_blocks=64, block_size=16)
        k1 = block_keys_from_tokens(list(range(32)), 16)
        first = store.insert_keys(k1, req_id="1")
        self.assertEqual(first["num_tokens"], 32)
        k2 = block_keys_from_tokens(list(range(48)), 16)
        second = store.insert_keys(k2, req_id="2")
        self.assertEqual(second["num_tokens"], 16)  # one new block
        self.assertEqual(second["num_blocks"], 1)
        cached = store.confirm_cached(k2, req_id="3")
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["num_tokens"], 0)

    def test_ssd_offload_hit_tier(self) -> None:
        store = MooncakeKvStore(num_blocks=1, block_size=16, num_ssd_blocks=2)
        k_a = block_keys_from_tokens(list(range(16)), 16)
        k_b = block_keys_from_tokens(list(range(16, 32)), 16)
        self.assertTrue(store.insert_keys(k_a, req_id="1")["ok"])
        self.assertTrue(store.insert_keys(k_b, req_id="2")["ok"])
        # A should have been offloaded to SSD.
        hit = store.lookup_keys(k_a, req_id="3", tokens_per_block=16, input_length=16)
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["tier"], "ssd")

    def test_ssd_staging_duration(self) -> None:
        owner = MagicMock()
        owner.sim.now.return_value = 0.0
        engine = MagicMock()
        client = KvClient(
            owner,
            store=None,
            engine=engine,
            bandwidth_gbps=50.0,
            bytes_per_token=16.0,
            transfer_s_floor=0.0,
            kv_latency_s=0.0,
            ssd_bandwidth_gbps=6.0,
            ssd_latency_s=0.001,
            on_transfer_complete=lambda *a: None,
        )
        tokens = 16
        net = transfer_duration_s(
            num_tokens=tokens,
            bytes_per_token_fallback=16.0,
            bandwidth_gbps=50.0,
            latency_s=0.0,
            transfer_s_floor=0.0,
        )
        staging = transfer_duration_s(
            num_tokens=tokens,
            bytes_per_token_fallback=16.0,
            bandwidth_gbps=6.0,
            latency_s=0.001,
            transfer_s_floor=0.0,
        )
        dram = client.transfer_duration_s(tokens, tier="dram")
        ssd = client.transfer_duration_s(tokens, tier="ssd")
        self.assertAlmostEqual(dram, net, places=9)
        self.assertAlmostEqual(ssd, staging + net, places=9)


if __name__ == "__main__":
    unittest.main()
