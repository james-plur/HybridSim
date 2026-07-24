"""KV Store / KV Client stubs (not wired into NO_NETWORK builder by default)."""

from __future__ import annotations

from typing import Any

from hybridsim import ActorBase, on

from hybridsim_infer.messages import KVLookupMsg, KVUpdateMsg


class KvStoreActor(ActorBase):
    """Stub remote KV store — handlers are no-ops / empty replies."""

    def __init__(self, *, sim, hs_actor, message_types: dict[str, Any]) -> None:
        self._store: dict[tuple[int, ...], Any] = {}
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    @on(KVLookupMsg)
    def on_lookup(self, _actor, msg: KVLookupMsg) -> None:
        key = tuple(msg.token_ids)
        hit = key in self._store
        try:
            self.reply({"hit": hit})
        except Exception:
            pass

    @on(KVUpdateMsg)
    def on_update(self, _actor, msg: KVUpdateMsg) -> None:
        key = tuple(msg.token_ids)
        self._store[key] = msg.request_id
        try:
            self.reply({"ok": True})
        except Exception:
            pass


class KvClientEngineActor(ActorBase):
    """Stub KV client engine — receives WorkloadMsg-style logic later."""

    def __init__(self, *, sim, hs_actor, message_types: dict[str, Any]) -> None:
        super().__init__(sim=sim, hs_actor=hs_actor, message_types=message_types)

    @on(KVLookupMsg)
    def on_lookup(self, _actor, msg: KVLookupMsg) -> None:
        pass

    @on(KVUpdateMsg)
    def on_update(self, _actor, msg: KVUpdateMsg) -> None:
        pass
