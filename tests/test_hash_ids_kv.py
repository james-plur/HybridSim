"""Unit tests for hash_ids-based Store / local APC (kvcache-simulator traces)."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("PYTHONHASHSEED", "0")

from hybridsim_infer.kv_system import (
    MooncakeKvStore,
    VllmKvCacheManager,
    block_keys_from_hash_ids,
    block_keys_from_tokens,
    prefix_hit_tokens,
    reset_none_hash,
    resolve_block_keys,
)
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators import KvCacheTraceRequestGenerator


class ResolveBlockKeysTests(unittest.TestCase):
    def test_hash_ids_preferred_over_tokens(self) -> None:
        keys = resolve_block_keys(
            token_ids=list(range(32)),
            hash_ids=[100, 200],
            block_size=16,
        )
        self.assertEqual(keys, ["100", "200"])

    def test_hash_ids_truncate_to_full_blocks(self) -> None:
        keys = resolve_block_keys(
            hash_ids=[1, 2, 3],
            block_size=16,
            num_tokens=32,
            input_length=40,
        )
        self.assertEqual(keys, ["1", "2"])

    def test_hash_ids_include_partial_last_on_full_prompt(self) -> None:
        keys = resolve_block_keys(
            hash_ids=[1, 2, 3],
            block_size=16,
            num_tokens=40,
            input_length=40,
        )
        self.assertEqual(keys, ["1", "2", "3"])

    def test_token_fallback_matches_vllm_chain(self) -> None:
        reset_none_hash()
        tokens = list(range(32))
        self.assertEqual(
            resolve_block_keys(token_ids=tokens, block_size=16),
            block_keys_from_tokens(tokens, 16),
        )

    def test_prefix_hit_tokens_partial_last(self) -> None:
        self.assertEqual(prefix_hit_tokens(2, 40, 16), 32)
        self.assertEqual(prefix_hit_tokens(3, 40, 16), 40)


class LocalApcHashIdsTests(unittest.TestCase):
    def test_match_and_cache_on_hash_chain(self) -> None:
        kv = VllmKvCacheManager(num_gpu_blocks=64, block_size=16)
        r1 = InferenceRequest(
            request_id=1,
            num_prefill_tokens=40,
            num_decode_tokens=1,
            prompt_token_ids=[],
            hash_ids=[10, 20, 30],
            block_size=16,
        )
        self.assertEqual(kv.match(r1), 0)
        r1.num_computed_tokens = 40
        kv.cache_request_prefix(r1)

        r2 = InferenceRequest(
            request_id=2,
            num_prefill_tokens=40,
            num_decode_tokens=1,
            prompt_token_ids=[],
            hash_ids=[10, 20, 99],
            block_size=16,
        )
        self.assertEqual(kv.match(r2), 32)

        r3 = InferenceRequest(
            request_id=3,
            num_prefill_tokens=40,
            num_decode_tokens=1,
            prompt_token_ids=[],
            hash_ids=[10, 20, 30],
            block_size=16,
        )
        # vLLM caps APC at prompt_len-1 then block-aligns → 32, not full 40.
        self.assertEqual(kv.match(r3), 32)


class StoreHashIdsTests(unittest.TestCase):
    def test_store_lookup_uses_hash_ids_and_trace_block_size(self) -> None:
        store = MooncakeKvStore(num_blocks=64, block_size=16)
        keys = block_keys_from_hash_ids([7, 8, 9])
        put = store.insert_keys(keys, req_id="1")
        self.assertTrue(put["ok"])

        hit = store.lookup_keys(
            keys,
            req_id="2",
            tokens_per_block=512,
            input_length=1200,
        )
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["num_blocks"], 3)
        self.assertEqual(hit["num_tokens"], 1200)

        partial = store.lookup_keys(
            block_keys_from_hash_ids([7, 8, 99]),
            req_id="3",
            tokens_per_block=512,
            input_length=1200,
        )
        self.assertEqual(partial["num_blocks"], 2)
        self.assertEqual(partial["num_tokens"], 1024)


class TraceGeneratorHashIdsTests(unittest.TestCase):
    def test_generator_preserves_hash_ids(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text(
                '{"timestamp":1.5,"block_size":64,"hash_ids":[1,2],"input_length":100,'
                '"output_length":8}\n',
                encoding="utf-8",
            )
            reqs = KvCacheTraceRequestGenerator(
                path, block_size=64, synthesize_prompt_tokens=True
            ).generate()
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].hash_ids, [1, 2])
        self.assertEqual(reqs[0].block_size, 64)
        self.assertEqual(reqs[0].num_prefill_tokens, 100)
        self.assertEqual(reqs[0].arrived_at, 1.5)


if __name__ == "__main__":
    unittest.main()
