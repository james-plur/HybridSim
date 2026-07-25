"""KV store backends: pure pool semantics shared by DES Actor and offline drivers.

``KvStoreActor`` only handles messages; CRUD / LRU live in ``MooncakeKvStore``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Optional

from hybridsim_infer.kv_system.block_keys import block_keys_from_tokens

PoolEventFn = Callable[..., None]


class KvStoreBackend(ABC):
    """Method-level KV pool API (exist / put / get / evict profile)."""

    block_size: int
    num_blocks: int

    @abstractmethod
    def lookup_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
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
    """Mooncake-master-like metadata registry: block keys + global capacity + LRU."""

    def __init__(
        self,
        *,
        num_blocks: int = 4096,
        block_size: int = 16,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.free_blocks = self.num_blocks
        self._blocks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._next_location = 1
        self._profile_fn = profile_fn
        self._profile_step_fn = profile_step_fn

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

    def snapshot_hashes(self) -> list[str]:
        return list(self._blocks.keys())

    def contains_all(self, keys: list[str]) -> bool:
        return bool(keys) and all(k in self._blocks for k in keys)

    def seed(self, token_ids: list[int]) -> None:
        keys = block_keys_from_tokens(token_ids, self.block_size)
        self.insert_keys(keys, req_id="seed")

    def lookup_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        hit_blocks = 0
        hit_mask: list[bool] = []
        for key in keys:
            present = key in self._blocks
            hit_mask.append(present)
            if not present:
                break
            self._blocks.move_to_end(key)
            hit_blocks += 1
        num_tokens = hit_blocks * self.block_size
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
        need = sum(1 for k in keys if k not in self._blocks)
        while need > self.free_blocks and self._blocks:
            drop_key, _ = self._blocks.popitem(last=False)
            self.free_blocks += 1
            self._emit("evict", [drop_key], req_id=req_id)
            need = sum(1 for k in keys if k not in self._blocks)
        if need > self.free_blocks:
            return {"ok": False, "reason": "oom"}
        location = self._next_location
        self._next_location += 1
        inserted: list[str] = []
        for key in keys:
            if key in self._blocks:
                self._blocks.move_to_end(key)
                continue
            self._blocks[key] = {
                "tokens_per_block": self.block_size,
                "location": location,
            }
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
            "num_tokens": len(keys) * self.block_size,
            "location": location,
            "num_blocks": len(keys),
        }

    def confirm_cached(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        for k in keys:
            self._blocks.move_to_end(k)
        self._emit(
            "exist",
            keys,
            req_id=req_id,
            hit_mask=[True] * len(keys),
            num_tokens=len(keys) * self.block_size,
        )
        return {
            "ok": True,
            "num_tokens": len(keys) * self.block_size,
            "cached": True,
            "location": self._blocks[keys[-1]]["location"],
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
        return {"ok": True, "num_tokens": n, "num_blocks": len(keys)}
