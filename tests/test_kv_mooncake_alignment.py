"""Unit tests for KV / Mooncake management alignment (save gate, APC, DRAM LRU)."""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "0")

from hybridsim_infer.schedulers.vllm_schedule import VllmScheduler
from hybridsim_infer.kv_system import MooncakeKvStore, VllmKvCacheManager, block_keys_from_tokens
from hybridsim_infer.kv_system.block_keys import (
    coarsen_keys_for_store,
    reset_none_hash,
    resolve_store_block_size,
    store_block_factor,
)
from hybridsim_infer.request import InferenceRequest, RequestStatus


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


class StoreCapacityDramTests(unittest.TestCase):
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

    def test_dram_lru_evicts_coldest(self) -> None:
        store = MooncakeKvStore(num_blocks=1, block_size=16)
        k_a = block_keys_from_tokens(list(range(16)), 16)
        k_b = block_keys_from_tokens(list(range(16, 32)), 16)
        self.assertTrue(store.insert_keys(k_a, req_id="1")["ok"])
        self.assertTrue(store.insert_keys(k_b, req_id="2")["ok"])
        miss = store.lookup_keys(k_a, req_id="3", tokens_per_block=16, input_length=16)
        self.assertFalse(miss["hit"])
        hit = store.lookup_keys(k_b, req_id="4", tokens_per_block=16, input_length=16)
        self.assertTrue(hit["hit"])
        self.assertEqual(len(store.snapshot_hashes()), 1)


class StoreNBlockSizeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_none_hash()

    def test_coarsen_keeps_window_last_key(self) -> None:
        gpu = block_keys_from_tokens(list(range(64)), 16)
        self.assertEqual(len(gpu), 4)
        self.assertEqual(coarsen_keys_for_store(gpu, 1), gpu)
        self.assertEqual(coarsen_keys_for_store(gpu, 4), [gpu[3]])
        self.assertEqual(coarsen_keys_for_store(gpu[:3], 4), [])
        self.assertEqual(store_block_factor(16, 64), 4)
        self.assertEqual(resolve_store_block_size(16, None), 16)
        with self.assertRaises(ValueError):
            resolve_store_block_size(16, 24)

    def test_lookup_n4_last_hash_is_full_window(self) -> None:
        gpu = block_keys_from_tokens(list(range(64)), 16)
        store = MooncakeKvStore(
            num_blocks=8, block_size=64, gpu_block_size=16
        )
        store.insert_keys([gpu[3]], req_id="seed")
        hit = store.lookup_keys(
            coarsen_keys_for_store(gpu, 4),
            req_id="1",
            tokens_per_block=64,
            input_length=64,
        )
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["num_tokens"], 64)
        self.assertEqual(hit["num_blocks"], 1)
        miss = store.lookup_keys(
            coarsen_keys_for_store(gpu[:3], 4),
            req_id="2",
            tokens_per_block=64,
            input_length=48,
        )
        self.assertFalse(miss["hit"])
        self.assertEqual(miss["num_tokens"], 0)

    def test_put_waits_for_aligned_window(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=64, block_size=16, store_block_size=64
        )
        gpu = block_keys_from_tokens(list(range(64)), 16)
        saved_keys: list[list[str]] = []
        pushes: list[int] = []

        class _Client:
            has_store = True

            async def save(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                keys = list(kwargs.get("block_keys") or [])
                saved_keys.append(keys)
                return {
                    "ok": True,
                    "num_tokens": len(keys) * 64,
                    "num_blocks": len(keys),
                    "cached": False,
                }

            def submit_push(self, rid: int, n: int) -> None:
                pushes.append(n)

        kv.attach_client(_Client())  # type: ignore[arg-type]
        req = InferenceRequest(
            request_id=3,
            num_prefill_tokens=80,
            num_decode_tokens=1,
            prompt_token_ids=list(range(80)),
            status=RequestStatus.RUNNING,
            num_computed_tokens=48,
        )
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(pushes, [])
        self.assertEqual(kv._num_saved_tokens.get(3, 0), 0)

        req.num_computed_tokens = 64
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(saved_keys, [[gpu[3]]])
        self.assertEqual(pushes, [64])
        self.assertEqual(kv._num_saved_tokens[3], 64)

        req.num_computed_tokens = 80
        asyncio.run(kv.save_computed_prefixes([req]))
        self.assertEqual(len(saved_keys), 1)
        self.assertEqual(kv._num_saved_tokens[3], 64)

    def test_get_scatter_block_ids_match_allocated(self) -> None:
        kv = VllmKvCacheManager(
            num_gpu_blocks=32, block_size=16, store_block_size=64
        )
        fw = VllmScheduler(
            tokens_per_step=64,
            reserve_full_isl=False,
            enable_prefix_caching=False,
        )
        req = InferenceRequest(
            request_id=9,
            num_prefill_tokens=64,
            num_decode_tokens=1,
            prompt_token_ids=list(range(64)),
            status=RequestStatus.WAITING,
        )

        async def remote_lookup(_request: InferenceRequest) -> dict[str, Any]:
            return {"hit": True, "num_tokens": 64}

        result = asyncio.run(
            fw.process_wait_queue(
                [req],
                [],
                kv_cache_manager=kv,
                token_budget=64,
                max_num_running_reqs=8,
                remote_lookup=remote_lookup,
            )
        )
        remote_pulls = result[5]
        self.assertEqual(len(remote_pulls), 1)
        self.assertEqual(len(remote_pulls[0].block_ids), 4)
        self.assertEqual(remote_pulls[0].num_tokens, 64)
        allocated = kv.allocated[9]
        self.assertEqual(
            remote_pulls[0].block_ids, [b.block_id for b in allocated]
        )


if __name__ == "__main__":
    unittest.main()
