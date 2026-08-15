"""KV store backends: pure pool semantics shared by DES Actor and offline drivers.

``KvStoreActor`` only handles messages; CRUD / LRU DRAM pool live in
``MooncakeKvStore``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Optional

from hybridsim_infer.kv_system.block_keys import (
    block_keys_from_tokens,
    coarsen_keys_for_store,
    prefix_hit_tokens,
    store_block_factor,
)

PoolEventFn = Callable[..., None]


class KvStoreBackend(ABC):
    """Method-level KV pool API (exist / put / get / evict profile)."""

    block_size: int
    num_blocks: int

    @abstractmethod
    def lookup_keys(
        self,
        keys: list[str],
        *,
        req_id: str = "",
        tokens_per_block: int = 0,
        input_length: int = 0,
    ) -> dict[str, Any]:
        """Longest contiguous prefix hit. Emits ``exist``."""

    @abstractmethod
    def insert_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        """Insert missing keys; may ``evict`` then ``put``. Returns status dict."""

    @abstractmethod
    def get_keys(
        self,
        keys: list[str],
        *,
        req_id: str = "",
        num_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Profile a remote get (pool contents unchanged)."""

    @abstractmethod
    def contains_all(self, keys: list[str]) -> bool:
        ...

    @abstractmethod
    def confirm_cached(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        """Idempotent hit path: LRU touch + ``exist`` profile."""

    @abstractmethod
    def snapshot_hashes(self) -> list[str]:
        ...

    @abstractmethod
    def seed(self, token_ids: list[int]) -> None:
        ...

    @abstractmethod
    def set_profile(
        self,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        ...


class MooncakeKvStore(KvStoreBackend):
    """Mooncake-like registry: DRAM block keys with LRU.

    Capacity:
    - ``num_blocks <= 0``: unlimited DRAM (no eviction).
    - DRAM full: evict coldest keys.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 4096,
        block_size: int = 16,
        gpu_block_size: int | None = None,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        #: Tokens per Store object (may be N × GPU page).
        self.block_size = int(block_size)
        #: GPU hash / page unit. Keys are coarsened from this chain.
        self.gpu_block_size = (
            int(gpu_block_size) if gpu_block_size is not None else int(block_size)
        )
        self.store_factor = store_block_factor(self.gpu_block_size, self.block_size)
        self._unlimited_dram = self.num_blocks <= 0
        self.free_blocks = 10**18 if self._unlimited_dram else self.num_blocks
        self._dram: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._next_location = 1
        self._profile_fn = profile_fn
        self._profile_step_fn = profile_step_fn

    # Back-compat alias used by actors / tests.
    @property
    def _blocks(self) -> OrderedDict[str, dict[str, Any]]:
        return self._dram

    def set_profile(
        self,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self._profile_fn = profile_fn
        self._profile_step_fn = profile_step_fn

    def _step(self) -> int:
        if self._profile_step_fn is not None:
            return int(self._profile_step_fn())
        return 0

    def _emit(self, op: str, keys: list[str], **kwargs: Any) -> None:
        if self._profile_fn is None:
            return
        self._profile_fn(
            op,
            keys=list(keys),
            hashes=list(keys),
            step=self._step(),
            **kwargs,
        )

    def _has(self, key: str) -> bool:
        return key in self._dram

    def snapshot_hashes(self) -> list[str]:
        return list(self._dram.keys())

    def contains_all(self, keys: list[str]) -> bool:
        return bool(keys) and all(self._has(k) for k in keys)

    def seed(self, token_ids: list[int]) -> None:
        gpu_keys = block_keys_from_tokens(token_ids, self.gpu_block_size)
        keys = coarsen_keys_for_store(gpu_keys, self.store_factor)
        self.insert_keys(keys, req_id="seed")

    def _evict_lru(self, req_id: str = "") -> bool:
        """Drop the coldest DRAM key."""
        if not self._dram:
            return False
        drop_key, _ = self._dram.popitem(last=False)
        self.free_blocks += 1
        self._emit("evict", [drop_key], req_id=req_id)
        return True

    def lookup_keys(
        self,
        keys: list[str],
        *,
        req_id: str = "",
        tokens_per_block: int = 0,
        input_length: int = 0,
    ) -> dict[str, Any]:
        hit_blocks = 0
        hit_mask: list[bool] = []
        hit_keys: list[str] = []
        for key in keys:
            present = self._has(key)
            hit_mask.append(present)
            if not present:
                break
            self._dram.move_to_end(key)
            hit_keys.append(key)
            hit_blocks += 1
        tpb = int(tokens_per_block) if int(tokens_per_block) > 0 else self.block_size
        num_tokens = prefix_hit_tokens(hit_blocks, int(input_length or 0), tpb)
        self._emit(
            "exist",
            keys[: max(hit_blocks, 0)],
            req_id=req_id,
            hit_mask=hit_mask,
            num_tokens=num_tokens,
        )
        return {
            "hit": hit_blocks > 0,
            "num_tokens": num_tokens,
            "num_blocks": hit_blocks,
        }

    def insert_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        need = sum(1 for k in keys if not self._has(k))
        while need > self.free_blocks and self._dram:
            if not self._evict_lru(req_id=req_id):
                break
            need = sum(1 for k in keys if not self._has(k))
        if need > self.free_blocks:
            return {"ok": False, "reason": "oom", "num_tokens": 0, "num_blocks": 0}

        location = self._next_location
        self._next_location += 1
        inserted: list[str] = []
        for key in keys:
            if key in self._dram:
                self._dram.move_to_end(key)
                continue
            self._dram[key] = {
                "tokens_per_block": self.block_size,
                "location": location,
            }
            if not self._unlimited_dram:
                self.free_blocks -= 1
            inserted.append(key)
        if inserted:
            self._emit(
                "put",
                inserted,
                req_id=req_id,
                num_tokens=len(inserted) * self.block_size,
            )
        return {
            "ok": True,
            "num_tokens": len(inserted) * self.block_size,
            "location": location,
            "num_blocks": len(inserted),
            "inserted_keys": list(inserted),
            "cached": False,
        }

    def confirm_cached(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        for k in keys:
            if not self._has(k):
                return {"ok": False, "reason": "missing", "num_tokens": 0, "cached": False}
            self._dram.move_to_end(k)
        self._emit(
            "exist",
            keys,
            req_id=req_id,
            hit_mask=[True] * len(keys),
            num_tokens=0,
        )
        loc = None
        last = keys[-1]
        if last in self._dram:
            loc = self._dram[last]["location"]
        return {
            "ok": True,
            "num_tokens": 0,
            "num_blocks": 0,
            "cached": True,
            "location": loc,
        }

    def get_keys(
        self,
        keys: list[str],
        *,
        req_id: str = "",
        num_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        n = (
            int(num_tokens)
            if num_tokens is not None
            else len(keys) * self.block_size
        )
        self._emit("get", list(keys), req_id=req_id, num_tokens=n)
        return {
            "ok": True,
            "num_tokens": n,
            "num_blocks": len(keys),
        }
