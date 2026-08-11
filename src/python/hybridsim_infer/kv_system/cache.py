"""Replica-local KV cache managers (GPU blocks + optional remote client)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from hybridsim_infer.kv_system.block_keys import (
    prefix_hit_tokens,
    resolve_block_keys,
)
from hybridsim_infer.kv_system.client import KvClient
from hybridsim_infer.request import InferenceRequest, RequestStatus


@dataclass
class KvBlock:
    block_id: int
    token_count: int = 0


class KvCacheManager(ABC):
    """Engine-agnostic local KV capacity API; optional remote via ``attach_client``."""

    block_size: int
    num_gpu_blocks: int
    free_blocks: int
    allocated: dict[int, list[KvBlock]]

    @abstractmethod
    def match(self, request: Any) -> int:
        """Longest local prefix-cache hit length (tokens)."""

    @abstractmethod
    def cache_prefix(self, token_ids: list[int]) -> None:
        ...

    def cache_request_prefix(self, request: Any) -> None:
        """Cache completed prompt prefix for a request (hash_ids or tokens)."""
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        n = int(
            getattr(request, "num_computed_tokens", 0)
            or getattr(request, "num_prefill_tokens", 0)
            or 0
        )
        if n > 0 and tokens:
            self.cache_prefix(tokens[:n] if n < len(tokens) else tokens)
        elif tokens:
            self.cache_prefix(tokens)

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

    # --- optional remote (default: disabled) ---

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


@dataclass
class VllmKvCacheManager(KvCacheManager):
    """vLLM-like GPU KV blocks + prefix cache; may hold a ``KvClient`` for remote.

    Local APC (block-hash table, contiguous prefix):
    - Prefer request ``hash_ids`` (trace / precomputed block keys).
    - Otherwise use ``resolve_block_keys`` / vLLM-compatible token block hashes
      (``PYTHONHASHSEED=0`` for alignment with offline vLLM hasher).
    """

    num_gpu_blocks: int = 1024
    block_size: int = 16
    free_blocks: int = 1024
    allocated: dict[int, list[KvBlock]] = field(default_factory=dict)
    _prefix_hash_chains: list[list[str]] = field(default_factory=list)
    _next_block_id: int = 0
    _client: Optional[KvClient] = field(default=None, repr=False)
    _kv_lookup_async: bool = False
    _pending_kv_pulls: set[int] = field(default_factory=set)
    _null_reserved: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._null_reserved = 1 if self.num_gpu_blocks > 0 else 0
        self.free_blocks = max(0, self.num_gpu_blocks - self._null_reserved)

    def _request_block_size(self, request: Any) -> int:
        rbs = int(getattr(request, "block_size", 0) or 0)
        return rbs if rbs > 0 else self.block_size

    # --- local ---

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
        """Block-hash keys for local APC (vLLM-compatible when no hash_ids)."""
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

    def match(self, request: Any) -> int:
        """Longest contiguous cached prefix in tokens (APC block-hash table).

        Mirrors vLLM ``get_computed_blocks``: hits are full blocks only, and
        ``max_cache_hit_length = num_prompt_tokens - 1`` so the last token is
        always recomputed for logits (may drop a whole trailing block).
        """
        bs = self._request_block_size(request)
        input_len = int(getattr(request, "num_prefill_tokens", 0) or 0)
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        if input_len <= 0 and tokens:
            input_len = len(tokens)
        keys = self._prefix_keys_for_request(request, num_tokens=input_len or None)
        if not keys or bs <= 0 or input_len <= 0:
            return 0
        best_blocks = 0
        for cached in self._prefix_hash_chains:
            n = 0
            lim = min(len(cached), len(keys))
            while n < lim and cached[n] == keys[n]:
                n += 1
            if n > best_blocks:
                best_blocks = n
        hit = prefix_hit_tokens(best_blocks, input_len, bs)
        # vLLM: never treat the full prompt as cached (need last token logits).
        max_hit = max(0, input_len - 1)
        hit = min(hit, max_hit)
        # allocate_slots expects block-aligned computed tokens for APC hits.
        return (hit // bs) * bs

    def cache_prefix(self, token_ids: list[int]) -> None:
        """Legacy helper: hash ``token_ids`` with manager ``block_size`` into APC."""
        if not token_ids:
            return
        keys = resolve_block_keys(
            token_ids=list(token_ids),
            block_size=self.block_size,
            num_tokens=len(token_ids),
            input_length=len(token_ids),
        )
        if not keys:
            return
        for cached in self._prefix_hash_chains:
            if cached == keys:
                return
        self._prefix_hash_chains.append(keys)

    def cache_request_prefix(self, request: Any) -> None:
        n = int(
            getattr(request, "num_computed_tokens", 0)
            or getattr(request, "num_prefill_tokens", 0)
            or 0
        )
        keys = self._prefix_keys_for_request(request, num_tokens=n if n > 0 else None)
        if not keys:
            return
        for cached in self._prefix_hash_chains:
            if cached == keys:
                return
        self._prefix_hash_chains.append(keys)

    def blocks_needed_to_hold(self, request: Any, num_tokens: int) -> int:
        rid = getattr(request, "request_id", id(request))
        need_blocks = self.blocks_for_tokens(int(num_tokens))
        have = len(self.allocated.get(rid, []))
        return max(0, need_blocks - have)

    def can_fit(
        self, request: Any, num_tokens: int, *, reserved_blocks: int = 0
    ) -> bool:
        grow = self.blocks_needed_to_hold(request, num_tokens)
        return grow <= max(0, self.free_blocks - reserved_blocks)

    def allocate(self, request: Any, num_tokens: int) -> Optional[list[KvBlock]]:
        if num_tokens < 0:
            return None
        rid = getattr(request, "request_id", id(request))
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        need_tokens = computed + int(num_tokens)
        grow = self.blocks_needed_to_hold(request, need_tokens)
        if grow <= 0:
            return []
        if grow > self.free_blocks:
            return None
        blocks: list[KvBlock] = []
        for _ in range(grow):
            bid = self._next_block_id
            self._next_block_id += 1
            blocks.append(KvBlock(block_id=bid, token_count=self.block_size))
        self.free_blocks -= grow
        self.allocated.setdefault(rid, []).extend(blocks)
        return blocks

    def free(self, request: Any) -> None:
        rid = getattr(request, "request_id", id(request))
        blocks = self.allocated.pop(rid, [])
        self.free_blocks += len(blocks)

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

        # Prefill-phase PD request: no remote lookup on this replica.
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

        # Decode-phase PD: skip Store hash match; control-plane RTT to source P.
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
            self._client.after_alloc_load(
                pull.request.request_id,
                pull.num_tokens,
                local_block_ids=None,
            )

    def on_lookup_reply(self, msg: Any) -> dict[str, Any]:
        if self._client is not None:
            return self._client.cache_lookup_reply(msg)
        return super().on_lookup_reply(msg)

    async def save_computed_prefixes(self, requests: list[InferenceRequest]) -> None:
        # Save when Store is wired (client may exist for PD control-plane alone).
        if self._client is None or not self._client.has_store:
            return
        for req in requests:
            # Prefill-phase PD handoff requests can still publish to Store.
            n = int(req.num_computed_tokens or 0)
            if n <= 0:
                continue
            tokens = list(req.prompt_token_ids[:n]) if req.prompt_token_ids else []
            hashes = list(req.hash_ids or [])
            if not tokens and not hashes:
                continue
            bs = self._request_block_size(req)
            input_len = int(req.num_prefill_tokens or 0)
            reply = await self._client.save(
                req.request_id,
                tokens,
                hash_ids=hashes or None,
                block_size=bs,
                num_tokens=n,
                input_length=input_len if input_len > 0 else n,
            )
            if reply and reply.get("ok") and not req.completed:
                n_tok = int(reply.get("num_tokens", 0))
                if n_tok > 0:
                    self._client.submit_push(req.request_id, n_tok)

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
            break
