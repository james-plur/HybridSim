"""KvStoreActor: DES facade over a shared ``KvStoreBackend`` (Mooncake by default).

Lives under ``actors/``; pool CRUD is in ``kv_system.store_backend``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.kv_system.block_keys import (
    block_aligned_tokens,
    block_keys_from_tokens,
    coarsen_keys_for_store,
    store_block_factor,
)
from hybridsim_infer.kv_system.store_backend import (
    KvStoreBackend,
    MooncakeKvStore,
    PoolEventFn,
)
from hybridsim_infer.messages import KVLookupMsg, KVLookupReplyMsg, KVUpdateMsg


class KvStoreActor(ActorBase):
    """Remote KV master Actor: messages only; pool CRUD lives in ``self.store``."""

    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        num_blocks: int = 4096,
        block_size: int = 16,
        gpu_block_size: int | None = None,
        num_ssd_blocks: int = 0,
        store: Optional[KvStoreBackend] = None,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.store: KvStoreBackend = store or MooncakeKvStore(
            num_blocks=num_blocks,
            block_size=block_size,
            gpu_block_size=gpu_block_size,
            num_ssd_blocks=num_ssd_blocks,
            profile_fn=profile_fn,
            profile_step_fn=profile_step_fn,
        )
        if profile_fn is not None or profile_step_fn is not None:
            self.store.set_profile(profile_fn, profile_step_fn)
        self.num_blocks = int(getattr(self.store, "num_blocks", num_blocks))
        self.block_size = int(self.store.block_size)
        self.gpu_block_size = int(
            getattr(self.store, "gpu_block_size", gpu_block_size or block_size)
        )
        self.store_factor = int(getattr(self.store, "store_factor", 1) or 1)
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    @property
    def free_blocks(self) -> int:
        return int(getattr(self.store, "free_blocks", 0))

    def set_profile(
        self,
        profile_fn: Optional[PoolEventFn] = None,
        profile_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.store.set_profile(profile_fn, profile_step_fn)

    def seed(self, token_ids: list[int]) -> None:
        self.store.seed(token_ids)

    def snapshot_hashes(self) -> list[str]:
        return self.store.snapshot_hashes()

    def _lookup_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        return self.store.lookup_keys(keys, req_id=req_id)

    def _insert_keys(self, keys: list[str], *, req_id: str = "") -> dict[str, Any]:
        return self.store.insert_keys(keys, req_id=req_id)

    def _fallback_keys(self, token_ids: list[int]) -> list[str]:
        gpu_keys = block_keys_from_tokens(list(token_ids), self.gpu_block_size)
        n = int(getattr(self, "store_factor", 1) or 1)
        if n <= 1:
            try:
                n = store_block_factor(self.gpu_block_size, self.block_size)
            except ValueError:
                n = 1
        return coarsen_keys_for_store(gpu_keys, n)

    @on(KVLookupMsg)
    def on_lookup(self, _actor, msg: KVLookupMsg) -> None:
        keys = list(msg.block_keys) if msg.block_keys else self._fallback_keys(
            list(msg.token_ids)
        )
        tpb = int(getattr(msg, "tokens_per_block", 0) or 0)
        input_length = int(getattr(msg, "input_length", 0) or 0)
        if msg.token_ids and not msg.block_keys:
            aligned = block_aligned_tokens(len(msg.token_ids), self.gpu_block_size)
            max_gpu = aligned // self.gpu_block_size if self.gpu_block_size else 0
            gpu_keys = block_keys_from_tokens(list(msg.token_ids), self.gpu_block_size)[
                :max_gpu
            ]
            keys = coarsen_keys_for_store(gpu_keys, max(1, self.store_factor))
            tpb = self.block_size
            input_length = len(msg.token_ids)
        elif tpb <= 0:
            tpb = self.block_size
        result = self.store.lookup_keys(
            keys,
            req_id=str(msg.request_id),
            tokens_per_block=tpb,
            input_length=input_length,
        )
        if msg.async_reply and msg.reply_to is not None:
            msg.reply_to.send(
                KVLookupReplyMsg,
                request_id=int(msg.request_id),
                hit=bool(result.get("hit")),
                num_tokens=int(result.get("num_tokens", 0)),
                num_blocks=int(result.get("num_blocks", 0)),
                location=None,
                tier=result.get("tier"),
            )
            return
        self.reply(result)

    @on(KVUpdateMsg)
    def on_update(self, _actor, msg: KVUpdateMsg) -> None:
        keys = list(msg.block_keys) if msg.block_keys else self._fallback_keys(
            list(msg.token_ids)
        )
        if not keys:
            self.reply({"ok": False, "reason": "empty"})
            return
        if self.store.contains_all(keys):
            self.reply(
                self.store.confirm_cached(keys, req_id=str(msg.request_id))
            )
            return
        self.reply(self.store.insert_keys(keys, req_id=str(msg.request_id)))
