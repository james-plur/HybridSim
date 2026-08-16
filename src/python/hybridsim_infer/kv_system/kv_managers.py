"""Replica-local KV cache managers (GPU BlockPool APC + optional remote client)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from hybridsim_infer.kv_system.block_keys import (
    block_aligned_tokens,
    coarsen_keys_for_store,
    full_block_count,
    prefix_hit_tokens,
    resolve_block_keys,
    resolve_store_block_size,
    store_block_factor,
)
from hybridsim_infer.kv_system.client import KvClient
from hybridsim_infer.request import InferenceRequest, RequestStatus


@dataclass
class KvBlock:
    block_id: int
    token_count: int = 0
    ref_cnt: int = 0
    block_hash: Optional[str] = None


def _cdiv(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b) if b > 0 else 0


class KvCacheManager(ABC):
    """Engine-agnostic local KV capacity API; optional remote via ``attach_client``."""

    block_size: int
    num_gpu_blocks: int
    free_blocks: int
    allocated: dict[int, list[KvBlock]]
    enable_prefix_caching: bool = False

    @abstractmethod
    def match(self, request: Any) -> int:
        """Longest local prefix-cache hit length (tokens)."""

    @abstractmethod
    def cache_prefix(self, token_ids: list[int]) -> None:
        """Seed APC from a token list (allocate, bind hashes, free pages)."""

    @abstractmethod
    def cache_request_prefix(self, request: Any) -> None:
        """Publish a request's computed prefix into APC."""

    @abstractmethod
    def blocks_for_tokens(self, num_tokens: int) -> int:
        ...

    @abstractmethod
    def blocks_needed_to_hold(self, request: Any, num_tokens: int) -> int:
        ...

    @abstractmethod
    def can_fit(
        self, request: Any, num_tokens: int, *, reserved_blocks: int = 0
    ) -> bool:
        ...

    @abstractmethod
    def allocate(self, request: Any, num_tokens: int) -> Optional[list[KvBlock]]:
        ...

    @abstractmethod
    def free(self, request: Any) -> None:
        ...

    @abstractmethod
    def preempt(self, request: Any) -> None:
        ...

    @property
    def remote_enabled(self) -> bool:
        return False

    @property
    def client_busy(self) -> bool:
        return False

    def attach_client(
        self,
        client: KvClient,
        *,
        kv_lookup_async: bool = False,
    ) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no remote client")

    def start_client(self) -> None:
        return None

    def check_client_error(self) -> None:
        return None

    async def remote_lookup(self, request: InferenceRequest) -> dict[str, Any]:
        return {"hit": False, "num_tokens": 0}

    def submit_remote_pulls(self, pulls: Any) -> None:
        return None

    def on_lookup_reply(self, msg: Any) -> dict[str, Any]:
        return {
            "hit": bool(getattr(msg, "hit", False)),
            "num_tokens": int(getattr(msg, "num_tokens", 0) or 0),
            "num_blocks": int(getattr(msg, "num_blocks", 0) or 0),
            "location": getattr(msg, "location", None),
        }

    async def save_computed_prefixes(self, requests: list[InferenceRequest]) -> None:
        return None

    def apply_pull_complete(
        self, request_id: int, waiting: list[InferenceRequest]
    ) -> None:
        return None

    @abstractmethod
    def attach_cached_prefix(
        self, request: Any, num_tokens: int
    ) -> Optional[list[KvBlock]]:
        """Reuse APC blocks for a block-aligned local hit."""


@dataclass
class VllmKvCacheManager(KvCacheManager):
    """vLLM-like GPU BlockPool + APC; may hold a ``KvClient`` for remote Store.

    Local APC (hash → physical block, ref_cnt, mid-flight visibility):
    - Prefer request ``hash_ids``; else vLLM-compatible token block hashes.
    - ``num_gpu_blocks <= 0``: unlimited GPU pages (no eviction).
    """

    num_gpu_blocks: int = 1024
    block_size: int = 16
    #: Store object size in tokens (multiple of ``block_size``). Default: same as GPU.
    store_block_size: int | None = None
    free_blocks: int = 1024
    enable_prefix_caching: bool = False
    allocated: dict[int, list[KvBlock]] = field(default_factory=dict)
    _client: Optional[KvClient] = field(default=None, repr=False)
    _kv_lookup_async: bool = False
    _pending_kv_pulls: set[int] = field(default_factory=set)
    _null_reserved: int = field(default=0, init=False)
    _unlimited: bool = field(default=False, init=False)
    _blocks: list[KvBlock] = field(default_factory=list, init=False, repr=False)
    _free_queue: OrderedDict[int, KvBlock] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _hash_to_block: dict[str, KvBlock] = field(
        default_factory=dict, init=False, repr=False
    )
    _next_block_id: int = field(default=0, init=False)
    _num_saved_tokens: dict[int, int] = field(default_factory=dict, init=False)
    _prefix_seed_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.store_block_size = resolve_store_block_size(
            self.block_size, self.store_block_size
        )
        self.store_factor = store_block_factor(self.block_size, self.store_block_size)
        self._unlimited = int(self.num_gpu_blocks) <= 0
        if self._unlimited:
            self._null_reserved = 0
            self.free_blocks = 10**18
            self._blocks = []
            self._free_queue = OrderedDict()
            return
        self._null_reserved = 1 if self.num_gpu_blocks > 0 else 0
        usable = max(0, self.num_gpu_blocks - self._null_reserved)
        self.free_blocks = usable
        # Reserve block 0 as null (never in free queue).
        null = KvBlock(block_id=0, ref_cnt=1)
        self._blocks = [null]
        self._free_queue = OrderedDict()
        for i in range(1, self.num_gpu_blocks):
            blk = KvBlock(block_id=i)
            self._blocks.append(blk)
            self._free_queue[i] = blk
        self._next_block_id = self.num_gpu_blocks

    def _request_block_size(self, request: Any) -> int:
        rbs = int(getattr(request, "block_size", 0) or 0)
        return rbs if rbs > 0 else self.block_size

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def capacity_tokens(self, request: Any) -> int:
        rid = getattr(request, "request_id", id(request))
        return len(self.allocated.get(rid, [])) * self.block_size

    def _prefix_keys_for_request(
        self,
        request: Any,
        *,
        num_tokens: int | None = None,
    ) -> list[str]:
        bs = self._request_block_size(request)
        if bs <= 0:
            return []
        hash_ids = list(getattr(request, "hash_ids", None) or [])
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        input_len = int(getattr(request, "num_prefill_tokens", 0) or 0)
        if input_len <= 0 and tokens:
            input_len = len(tokens)
        n = int(num_tokens) if num_tokens is not None else input_len
        if hash_ids:
            return resolve_block_keys(
                hash_ids=hash_ids,
                block_size=bs,
                num_tokens=n if n > 0 else None,
                input_length=input_len if input_len > 0 else None,
            )
        if not tokens:
            return []
        return resolve_block_keys(
            token_ids=tokens,
            block_size=bs,
            num_tokens=n if n > 0 else None,
            input_length=input_len if input_len > 0 else None,
        )

    def _touch_block(self, block: KvBlock) -> None:
        if block.ref_cnt == 0 and block.block_id in self._free_queue:
            self._free_queue.pop(block.block_id, None)
            if not self._unlimited:
                self.free_blocks = max(0, self.free_blocks - 1)
        block.ref_cnt += 1

    def _return_block_to_free(self, block: KvBlock) -> None:
        if block.block_id == 0 and not self._unlimited:
            return
        block.ref_cnt = max(0, block.ref_cnt - 1)
        if block.ref_cnt > 0:
            return
        if self._unlimited:
            # Drop unreferenced unlimited blocks from hash map only when unhashed.
            if block.block_hash is None:
                return
            # Keep hashed blocks as eviction candidates in free queue.
        if block.block_id in self._free_queue:
            return
        if block.block_hash is None:
            # Evict first: prepend unhashed.
            items = list(self._free_queue.items())
            self._free_queue.clear()
            self._free_queue[block.block_id] = block
            for k, v in items:
                self._free_queue[k] = v
        else:
            self._free_queue[block.block_id] = block
        if not self._unlimited:
            self.free_blocks += 1

    def _evict_one_free(self) -> Optional[KvBlock]:
        if not self._free_queue:
            return None
        bid, block = self._free_queue.popitem(last=False)
        if not self._unlimited:
            self.free_blocks = max(0, self.free_blocks - 1)
        if block.block_hash is not None:
            self._hash_to_block.pop(block.block_hash, None)
            block.block_hash = None
        block.ref_cnt = 0
        return block

    def _alloc_fresh_blocks(self, n: int) -> Optional[list[KvBlock]]:
        if n <= 0:
            return []
        if self._unlimited:
            out: list[KvBlock] = []
            for _ in range(n):
                bid = self._next_block_id
                self._next_block_id += 1
                blk = KvBlock(block_id=bid, token_count=self.block_size, ref_cnt=1)
                out.append(blk)
            return out
        if n > self.free_blocks:
            return None
        out = []
        for _ in range(n):
            blk = self._evict_one_free()
            if blk is None:
                # Roll back
                for b in out:
                    self._return_block_to_free(b)
                return None
            blk.ref_cnt = 1
            blk.token_count = self.block_size
            out.append(blk)
        return out

    def match(self, request: Any) -> int:
        """Longest contiguous cached prefix in tokens (APC block-hash table)."""
        if not self.enable_prefix_caching:
            return 0
        bs = self._request_block_size(request)
        input_len = int(getattr(request, "num_prefill_tokens", 0) or 0)
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        if input_len <= 0 and tokens:
            input_len = len(tokens)
        keys = self._prefix_keys_for_request(request, num_tokens=input_len or None)
        if not keys or bs <= 0 or input_len <= 0:
            return 0

        n = 0
        for key in keys:
            if key not in self._hash_to_block:
                break
            n += 1

        hit = prefix_hit_tokens(n, input_len, bs)
        max_hit = max(0, input_len - 1)
        hit = min(hit, max_hit)
        return (hit // bs) * bs

    def attach_cached_prefix(
        self, request: Any, num_tokens: int
    ) -> Optional[list[KvBlock]]:
        """Reuse APC blocks for a block-aligned hit; add to ``allocated``."""
        bs = self._request_block_size(request)
        n = block_aligned_tokens(int(num_tokens), bs)
        if n <= 0:
            return []
        keys = self._prefix_keys_for_request(request, num_tokens=n)
        need = full_block_count(n, bs)
        if len(keys) < need:
            return None
        rid = getattr(request, "request_id", id(request))
        have = list(self.allocated.get(rid, []))
        if len(have) >= need:
            return []
        attached: list[KvBlock] = []
        for i in range(len(have), need):
            key = keys[i]
            block = self._hash_to_block.get(key)
            if block is None:
                for b in attached:
                    self._return_block_to_free(b)
                return None
            self._touch_block(block)
            attached.append(block)
        self.allocated.setdefault(rid, []).extend(attached)
        return attached

    def cache_prefix(self, token_ids: list[int]) -> None:
        """Seed APC from a token list: allocate, hash-bind, then free the pages."""
        tokens = list(token_ids)
        if not tokens or not self.enable_prefix_caching:
            return
        self._prefix_seed_id -= 1
        req = InferenceRequest(
            request_id=self._prefix_seed_id,
            num_prefill_tokens=len(tokens),
            prompt_token_ids=tokens,
        )
        if self.allocate(req, len(tokens)) is None:
            return
        self.free(req)

    def cache_request_prefix(self, request: Any) -> None:
        n = int(
            getattr(request, "num_computed_tokens", 0)
            or getattr(request, "num_prefill_tokens", 0)
            or 0
        )
        self.cache_blocks(request, n)

    def cache_blocks(self, request: Any, num_tokens: int) -> None:
        """Publish full blocks into APC (vLLM ``cache_blocks``)."""
        bs = self._request_block_size(request)
        rid = getattr(request, "request_id", id(request))
        blocks = self.allocated.get(rid, [])
        n_full = full_block_count(int(num_tokens), bs)
        keys = self._prefix_keys_for_request(
            request, num_tokens=n_full * bs if n_full > 0 else int(num_tokens) or None
        )
        if not self.enable_prefix_caching or n_full <= 0 or not blocks:
            return
        for i in range(min(n_full, len(blocks), len(keys))):
            block = blocks[i]
            key = keys[i]
            if block.block_hash == key:
                continue
            if block.block_hash is not None and self._hash_to_block.get(
                block.block_hash
            ) is block:
                self._hash_to_block.pop(block.block_hash, None)
            if key not in self._hash_to_block:
                block.block_hash = key
                self._hash_to_block[key] = block
            elif self._hash_to_block[key] is block:
                block.block_hash = key
            else:
                block.block_hash = None

    def blocks_needed_to_hold(self, request: Any, num_tokens: int) -> int:
        rid = getattr(request, "request_id", id(request))
        need_blocks = self.blocks_for_tokens(int(num_tokens))
        have = len(self.allocated.get(rid, []))
        return max(0, need_blocks - have)

    def can_fit(
        self, request: Any, num_tokens: int, *, reserved_blocks: int = 0
    ) -> bool:
        grow = self.blocks_needed_to_hold(request, num_tokens)
        if self._unlimited:
            return True
        return grow <= max(0, self.free_blocks - reserved_blocks)

    def allocate(self, request: Any, num_tokens: int) -> Optional[list[KvBlock]]:
        if num_tokens < 0:
            return None
        rid = getattr(request, "request_id", id(request))
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        need_tokens = computed + int(num_tokens)
        grow = self.blocks_needed_to_hold(request, need_tokens)
        if grow <= 0:
            if self.enable_prefix_caching:
                self.cache_blocks(request, need_tokens)
            return []
        fresh = self._alloc_fresh_blocks(grow)
        if fresh is None:
            return None
        self.allocated.setdefault(rid, []).extend(fresh)
        if self.enable_prefix_caching:
            self.cache_blocks(request, need_tokens)
        return fresh

    def free(self, request: Any) -> None:
        rid = getattr(request, "request_id", id(request))
        blocks = self.allocated.pop(rid, [])
        # Reverse order: tail first (vLLM eviction priority).
        for block in reversed(blocks):
            self._return_block_to_free(block)
        self._num_saved_tokens.pop(rid, None)

    def preempt(self, request: Any) -> None:
        self.free(request)
        request.num_computed_tokens = 0
        request.pending_remote_tokens = 0

    # --- remote ---

    @property
    def remote_enabled(self) -> bool:
        return self._client is not None

    @property
    def client_busy(self) -> bool:
        return bool(self._client and self._client.busy)

    def attach_client(
        self,
        client: KvClient,
        *,
        kv_lookup_async: bool = False,
    ) -> None:
        self._client = client
        self._kv_lookup_async = bool(kv_lookup_async)

    def start_client(self) -> None:
        if self._client is not None:
            self._client.start()

    def check_client_error(self) -> None:
        if self._client is not None:
            self._client.check_error()

    async def remote_lookup(self, request: InferenceRequest) -> dict[str, Any]:
        if self._client is None:
            return {"hit": False, "num_tokens": 0}
        params = request.kv_transfer_params or {}

        if params.get("do_remote_decode") and not params.get("do_remote_prefill"):
            return {"hit": False, "num_tokens": 0}

        cached = request.lookup_result
        if cached is None:
            cached = self._client.take_cached_lookup(request.request_id)
        if cached is not None:
            request.lookup_result = None
            request.pending_lookup = False
            return cached

        tokens = list(request.prompt_token_ids)
        hashes = list(request.hash_ids or [])
        bs = self._request_block_size(request)
        input_len = int(request.num_prefill_tokens or len(tokens) or 0)

        if params.get("do_remote_prefill"):
            if request.pending_lookup:
                return {"pending": True}
            location = params.get("remote_replica_id")
            self._client.lookup_control_plane(
                request.request_id,
                tokens,
                location=location,
                num_tokens=input_len,
                block_size=bs,
            )
            request.pending_lookup = True
            return {"pending": True}

        if self._kv_lookup_async:
            if request.pending_lookup:
                return {"pending": True}
            self._client.lookup_async(
                request.request_id,
                tokens,
                hash_ids=hashes or None,
                block_size=bs,
                num_tokens=input_len,
                input_length=input_len,
            )
            request.pending_lookup = True
            return {"pending": True}

        return await self._client.lookup(
            request.request_id,
            tokens,
            hash_ids=hashes or None,
            block_size=bs,
            num_tokens=input_len,
            input_length=input_len,
        )

    def submit_remote_pulls(self, pulls: Any) -> None:
        if self._client is None:
            return
        for pull in pulls:
            self._pending_kv_pulls.add(pull.request.request_id)
            block_ids = [
                int(x) for x in (getattr(pull, "block_ids", None) or [])
            ]
            self._client.after_alloc_load(
                pull.request.request_id,
                pull.num_tokens,
                local_block_ids=block_ids or None,
            )

    def on_lookup_reply(self, msg: Any) -> dict[str, Any]:
        if self._client is not None:
            return self._client.cache_lookup_reply(msg)
        return super().on_lookup_reply(msg)

    async def save_computed_prefixes(self, requests: list[InferenceRequest]) -> None:
        """Save newly completed full blocks to Store (vLLM Mooncake gate)."""
        if self._client is None or not self._client.has_store:
            return
        for req in requests:
            if req.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
                continue
            rid = int(req.request_id)
            bs = self._request_block_size(req)
            if bs <= 0:
                continue
            computed = int(req.num_computed_tokens or 0)
            prefill_end = int(req.num_prefill_tokens or 0)
            if prefill_end <= 0 and req.prompt_token_ids:
                prefill_end = len(req.prompt_token_ids)
            # Do not save decode-only suffix past prefill (vLLM scheduler gate).
            save_upto = min(computed, prefill_end) if prefill_end > 0 else computed
            if save_upto <= 0:
                continue
            n = int(getattr(self, "store_factor", 1) or 1)
            store_bs = int(getattr(self, "store_block_size", 0) or 0) or (n * bs)
            aligned_gpu = block_aligned_tokens(save_upto, bs)
            aligned_store = (aligned_gpu // store_bs) * store_bs if store_bs > 0 else 0
            saved = int(self._num_saved_tokens.get(rid, 0))
            if aligned_store <= saved:
                continue
            all_gpu_keys = self._prefix_keys_for_request(
                req, num_tokens=aligned_store
            )
            all_store_keys = coarsen_keys_for_store(all_gpu_keys, n)
            start_w = saved // store_bs if store_bs > 0 else 0
            end_w = aligned_store // store_bs if store_bs > 0 else 0
            keys = all_store_keys[start_w:end_w]
            if not keys:
                continue
            tokens = (
                list(req.prompt_token_ids[:aligned_store])
                if req.prompt_token_ids
                else []
            )
            hashes = list(req.hash_ids or [])
            reply = await self._client.save(
                rid,
                tokens,
                hash_ids=hashes or None,
                block_size=bs,
                num_tokens=aligned_store,
                input_length=prefill_end if prefill_end > 0 else aligned_store,
                block_keys=keys,
            )
            if not reply or not reply.get("ok"):
                continue
            self._num_saved_tokens[rid] = aligned_store
            # Only push newly inserted bytes (not confirm_cached).
            if reply.get("cached"):
                continue
            n_tok = int(reply.get("num_tokens", 0) or 0)
            if n_tok <= 0:
                n_tok = aligned_store - saved
            if n_tok > 0 and not req.completed:
                self._client.submit_push(rid, n_tok)

    def apply_pull_complete(
        self, request_id: int, waiting: list[InferenceRequest]
    ) -> None:
        rid = int(request_id)
        if rid not in self._pending_kv_pulls:
            return
        self._pending_kv_pulls.discard(rid)
        for req in waiting:
            if req.request_id != rid:
                continue
            if req.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
                hit = req.pending_remote_tokens
                req.num_computed_tokens = max(req.num_computed_tokens, hit)
                req.pending_remote_tokens = 0
                req.status = RequestStatus.WAITING
                if self.enable_prefix_caching:
                    self.cache_blocks(req, req.num_computed_tokens)
            break
