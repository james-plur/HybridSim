"""KV store backends: pure pool semantics shared by DES Actor and offline drivers.

``KvStoreActor`` only handles messages; CRUD / LRU / DRAM+SSD tier live in
``MooncakeKvStore``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Literal, Optional

from hybridsim_infer.kv_system.block_keys import (
    block_keys_from_tokens,
    prefix_hit_tokens,
)

PoolEventFn = Callable[..., None]
TierName = Literal["dram", "ssd"]


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
    """Mooncake-like registry: DRAM (+ optional SSD) block keys with LRU.

    Capacity:
    - ``num_blocks <= 0``: unlimited DRAM (no DRAM eviction).
    - ``num_ssd_blocks <= 0``: SSD tier disabled.
    - DRAM full + SSD enabled: cold DRAM keys offload to SSD.
    - SSD full: evict coldest SSD keys.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 4096,
        block_size: int = 16,
        num_ssd_blocks: int = 0,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_ssd_blocks = int(num_ssd_blocks)
        self._unlimited_dram = self.num_blocks <= 0
        self._ssd_enabled = self.num_ssd_blocks > 0
        self.free_blocks = 10**18 if self._unlimited_dram else self.num_blocks
        self.free_ssd_blocks = (
            10**18 if not self._ssd_enabled else self.num_ssd_blocks
        )
        self._dram: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._ssd: OrderedDict[str, dict[str, Any]] = OrderedDict()
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
        return key in self._dram or key in self._ssd

    def _tier_of(self, key: str) -> Optional[TierName]:
        if key in self._dram:
            return "dram"
        if key in self._ssd:
            return "ssd"
        return None

    def snapshot_hashes(self) -> list[str]:
        return list(self._dram.keys()) + list(self._ssd.keys())

    def contains_all(self, keys: list[str]) -> bool:
        return bool(keys) and all(self._has(k) for k in keys)

    def seed(self, token_ids: list[int]) -> None:
        keys = block_keys_from_tokens(token_ids, self.block_size)
        self.insert_keys(keys, req_id="seed")

    def _evict_ssd(self, req_id: str = "") -> bool:
        if not self._ssd:
            return False
        drop_key, _ = self._ssd.popitem(last=False)
        if not self._unlimited_dram:
            pass
        self.free_ssd_blocks += 1
        self._emit("evict", [drop_key], req_id=req_id, tier="ssd")
        return True

    def _offload_dram_to_ssd(self, req_id: str = "") -> bool:
        """Move coldest DRAM key to SSD (or drop if SSD disabled/full)."""
        if not self._dram:
            return False
        if not self._ssd_enabled:
            drop_key, _ = self._dram.popitem(last=False)
            self.free_blocks += 1
            self._emit("evict", [drop_key], req_id=req_id, tier="dram")
            return True
        while self.free_ssd_blocks <= 0 and self._ssd:
            self._evict_ssd(req_id=req_id)
        if self.free_ssd_blocks <= 0:
            drop_key, _ = self._dram.popitem(last=False)
            self.free_blocks += 1
            self._emit("evict", [drop_key], req_id=req_id, tier="dram")
            return True
        drop_key, meta = self._dram.popitem(last=False)
        self.free_blocks += 1
        meta = dict(meta)
        meta["tier"] = "ssd"
        self._ssd[drop_key] = meta
        self.free_ssd_blocks -= 1
        self._emit("offload", [drop_key], req_id=req_id, tier="ssd")
        return True

    def _promote_to_dram(self, key: str, req_id: str = "") -> None:
        if key not in self._ssd:
            return
        if not self._unlimited_dram and self.free_blocks <= 0:
            if not self._offload_dram_to_ssd(req_id=req_id):
                return
            if self.free_blocks <= 0:
                return
        meta = self._ssd.pop(key)
        self.free_ssd_blocks += 1
        meta = dict(meta)
        meta["tier"] = "dram"
        self._dram[key] = meta
        if not self._unlimited_dram:
            self.free_blocks -= 1
        self._emit("promote", [key], req_id=req_id, tier="dram")

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
        # Tier at hit time (before promote): SSD staging still applies after promote.
        worst_tier: Optional[TierName] = None
        for key in keys:
            tier = self._tier_of(key)
            present = tier is not None
            hit_mask.append(present)
            if not present:
                break
            if tier == "dram":
                self._dram.move_to_end(key)
                if worst_tier is None:
                    worst_tier = "dram"
            else:
                assert tier == "ssd"
                self._ssd.move_to_end(key)
                worst_tier = "ssd"
                # Mooncake-style: promote SSD hits back to DRAM when possible.
                self._promote_to_dram(key, req_id=req_id)
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
            tier=worst_tier if hit_blocks > 0 else None,
        )
        return {
            "hit": hit_blocks > 0,
            "num_tokens": num_tokens,
            "num_blocks": hit_blocks,
            "tier": worst_tier if hit_blocks > 0 else None,
        }

    def insert_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        need = sum(1 for k in keys if not self._has(k))
        while need > self.free_blocks and self._dram:
            if not self._offload_dram_to_ssd(req_id=req_id):
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
            if key in self._ssd:
                # Already on SSD: touch + promote if possible; not a new insert.
                self._ssd.move_to_end(key)
                self._promote_to_dram(key, req_id=req_id)
                continue
            self._dram[key] = {
                "tokens_per_block": self.block_size,
                "location": location,
                "tier": "dram",
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
                tier="dram",
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
        worst_tier: TierName = "dram"
        for k in keys:
            tier = self._tier_of(k)
            if tier is None:
                return {"ok": False, "reason": "missing", "num_tokens": 0, "cached": False}
            if tier == "dram":
                self._dram.move_to_end(k)
            else:
                self._ssd.move_to_end(k)
                worst_tier = "ssd"
                self._promote_to_dram(k, req_id=req_id)
        self._emit(
            "exist",
            keys,
            req_id=req_id,
            hit_mask=[True] * len(keys),
            num_tokens=0,
            tier=worst_tier,
        )
        loc = None
        last = keys[-1]
        if last in self._dram:
            loc = self._dram[last]["location"]
        elif last in self._ssd:
            loc = self._ssd[last]["location"]
        return {
            "ok": True,
            "num_tokens": 0,
            "num_blocks": 0,
            "cached": True,
            "location": loc,
            "tier": worst_tier,
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
        tier: Optional[TierName] = None
        for k in keys:
            t = self._tier_of(k)
            if t == "ssd":
                tier = "ssd"
                break
            if t == "dram" and tier is None:
                tier = "dram"
        self._emit("get", list(keys), req_id=req_id, num_tokens=n, tier=tier)
        return {
            "ok": True,
            "num_tokens": n,
            "num_blocks": len(keys),
            "tier": tier,
        }
