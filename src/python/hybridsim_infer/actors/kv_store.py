"""KV Store actor and KV client engine glue."""

from __future__ import annotations

from typing import Any, Callable, Optional

from hybridsim import ActorBase, on

from hybridsim_infer.messages import KVLookupMsg, KVUpdateMsg


class KvStoreActor(ActorBase):
    """Remote KV block store: sync lookup / update with simple capacity."""

    def __init__(
        self,
        *,
        sim,
        hs_actor,
        message_types: dict[str, Any],
        num_blocks: int = 4096,
        block_size: int = 16,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.free_blocks = self.num_blocks
        # prefix token tuple → stored token count
        self._entries: dict[tuple[int, ...], int] = {}
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    def seed(self, token_ids: list[int]) -> None:
        """Pre-populate store (for demos/tests)."""
        key = tuple(token_ids)
        need = max(1, (len(token_ids) + self.block_size - 1) // self.block_size)
        if key not in self._entries and need <= self.free_blocks:
            self._entries[key] = len(token_ids)
            self.free_blocks -= need

    def _longest_hit(self, token_ids: list[int]) -> int:
        if not token_ids:
            return 0
        best = 0
        for key, n in self._entries.items():
            lim = min(len(key), len(token_ids), n)
            matched = 0
            while matched < lim and key[matched] == token_ids[matched]:
                matched += 1
            if matched > best:
                best = matched
        return best

    def lookup(self, token_ids: list[int]) -> dict[str, Any]:
        """Synchronous longest-prefix hit (schedule_step remote_lookup)."""
        n = self._longest_hit(list(token_ids))
        return {"hit": n > 0, "num_tokens": n}

    @on(KVLookupMsg)
    def on_lookup(self, _actor, msg: KVLookupMsg) -> None:
        self.reply(self.lookup(list(msg.token_ids)))

    @on(KVUpdateMsg)
    def on_update(self, _actor, msg: KVUpdateMsg) -> None:
        token_ids = list(msg.token_ids)
        if not token_ids:
            self.reply({"ok": False, "reason": "empty"})
            return
        key = tuple(token_ids)
        if key in self._entries:
            self.reply({"ok": True, "num_tokens": len(token_ids), "cached": True})
            return
        need = max(1, (len(token_ids) + self.block_size - 1) // self.block_size)
        if need > self.free_blocks:
            # Simple eviction: drop arbitrary oldest-ish entry
            if self._entries:
                drop_key = next(iter(self._entries))
                drop_n = self._entries.pop(drop_key)
                freed = max(1, (drop_n + self.block_size - 1) // self.block_size)
                self.free_blocks += freed
            if need > self.free_blocks:
                self.reply({"ok": False, "reason": "oom"})
                return
        self._entries[key] = len(token_ids)
        self.free_blocks -= need
        self.reply({"ok": True, "num_tokens": len(token_ids), "location": len(self._entries)})


class KvClientEngine:
    """Wraps EngineActor for async KV transfers; completion → KVTransferEndMsg."""

    def __init__(
        self,
        engine,
        *,
        on_transfer_complete: Callable[[int, int], None],
    ) -> None:
        self._engine = engine
        self._on_transfer_complete = on_transfer_complete
        self._inflight: dict[int, int] = {}  # workload_id → request_id
        self._engine.set_on_workload_complete(self._handle_complete)

    @property
    def busy(self) -> bool:
        return bool(self._inflight)

    def start(self) -> None:
        self._engine.start()

    def check_error(self) -> None:
        self._engine.check_error()

    def submit(self, workload: dict[str, Any], request_id: int) -> None:
        wid = int(workload["workload_id"])
        self._inflight[wid] = int(request_id)
        self._engine.send_workload(workload)

    def _handle_complete(self, workload_id: int) -> None:
        rid = self._inflight.pop(int(workload_id), None)
        if rid is not None:
            self._on_transfer_complete(int(workload_id), int(rid))
