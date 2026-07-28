"""KvClient: replica-local Mooncake-style store client + transfer engine."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from hybridsim_infer.kv_system.block_keys import block_aligned_tokens, block_keys_from_tokens
from hybridsim_infer.messages import KVLookupMsg, KVLookupReplyMsg, KVUpdateMsg
from hybridsim_infer.workload_generators.kv_transfer import KvTransferWorkloadGenerator

TransferDirection = Literal["pull", "push"]


class KvClient:
    """Replica-held object (not an Actor): talks to KvStoreActor + owns a transfer Engine.

    Analogous to the local Mooncake client embedded in a vLLM instance:
    - sync/async metadata RPC to centralized master (``KvStoreActor``)
    - control-plane lookup (PD Decode): skip hash match, simulate Prefill-GPU RTT
    - async data plane via TimeoutKernel on a dedicated EngineActor
    """

    def __init__(
        self,
        owner: Any,
        store: Any,
        engine: Any,
        *,
        block_size: int = 16,
        bandwidth_gbps: float = 50.0,
        bytes_per_token: float = 16.0,
        transfer_s_floor: float = 1e-4,
        lookup_rtt_s: float = 1e-3,
        on_transfer_complete: Callable[[int, int, str], None],
        workload_generator: Optional[KvTransferWorkloadGenerator] = None,
        profile: Any = None,
        replica_id: int = 0,
    ) -> None:
        self._owner = owner
        self._store = store
        self._engine = engine
        self._profile = profile
        self._replica_id = int(replica_id)
        self.block_size = int(block_size)
        self.bandwidth_gbps = float(bandwidth_gbps)
        self.bytes_per_token = float(bytes_per_token)
        self.transfer_s_floor = float(transfer_s_floor)
        self.lookup_rtt_s = float(lookup_rtt_s)
        self._on_transfer_complete = on_transfer_complete
        self._workload_generator = workload_generator or KvTransferWorkloadGenerator()
        self._inflight: dict[int, tuple[int, TransferDirection]] = {}
        self._lookup_cache: dict[int, dict[str, Any]] = {}
        self._next_workload_id = 1
        self._engine.set_on_workload_complete(self._handle_complete)

    @property
    def busy(self) -> bool:
        return bool(self._inflight)

    @property
    def has_store(self) -> bool:
        return self._store is not None

    def start(self) -> None:
        self._engine.start()

    def check_error(self) -> None:
        self._engine.check_error()

    def transfer_duration_s(self, num_tokens: int) -> float:
        nbytes = max(0, int(num_tokens)) * self.bytes_per_token
        bps = max(1e-9, self.bandwidth_gbps) * (1e9 / 8.0)
        return max(self.transfer_s_floor, nbytes / bps)

    def control_plane_hit(
        self,
        token_ids: list[int],
        *,
        location: Any,
    ) -> dict[str, Any]:
        """Full prompt as hit without Store hash match (Prefill GPU known)."""
        num_tokens = block_aligned_tokens(len(token_ids), self.block_size)
        return {
            "hit": num_tokens > 0,
            "num_tokens": num_tokens,
            "num_blocks": num_tokens // self.block_size if self.block_size else 0,
            "location": location,
            "mode": "control_plane",
        }

    def lookup_control_plane(
        self,
        request_id: int,
        token_ids: list[int],
        *,
        location: Any,
    ) -> dict[str, Any]:
        """Fire delayed control-plane reply (RTT); caller treats as pending."""
        result = self.control_plane_hit(token_ids, location=location)
        self._owner.send(
            KVLookupReplyMsg,
            delay=self.lookup_rtt_s,
            request_id=int(request_id),
            hit=bool(result["hit"]),
            num_tokens=int(result["num_tokens"]),
            num_blocks=int(result["num_blocks"]),
            location=location,
        )
        return {"pending": True}

    def take_cached_lookup(self, request_id: int) -> Optional[dict[str, Any]]:
        return self._lookup_cache.pop(int(request_id), None)

    def cache_lookup_reply(self, msg: KVLookupReplyMsg) -> dict[str, Any]:
        result = {
            "hit": bool(msg.hit),
            "num_tokens": int(msg.num_tokens),
            "num_blocks": int(msg.num_blocks),
            "location": msg.location,
            "mode": "control_plane" if msg.location is not None else "store",
        }
        self._lookup_cache[int(msg.request_id)] = result
        return result

    async def lookup(self, request_id: int, token_ids: list[int]) -> dict[str, Any]:
        """Sync metadata lookup on master (block-aligned hit length)."""
        if self._store is None:
            return {"hit": False, "num_tokens": 0}
        keys = block_keys_from_tokens(token_ids, self.block_size)
        return await self._owner.request(
            self._store,
            KVLookupMsg,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
            async_reply=False,
            reply_to=None,
        )

    def lookup_async(self, request_id: int, token_ids: list[int]) -> None:
        """Fire-and-forget lookup; result arrives as ``KVLookupReplyMsg`` on owner."""
        if self._store is None:
            # No store: synthesize an immediate miss reply on the owner.
            self._owner.send(
                KVLookupReplyMsg,
                delay=self.lookup_rtt_s,
                request_id=int(request_id),
                hit=False,
                num_tokens=0,
                num_blocks=0,
            )
            return
        keys = block_keys_from_tokens(token_ids, self.block_size)
        self._store.send(
            KVLookupMsg,
            delay=self.lookup_rtt_s,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
            async_reply=True,
            reply_to=self._owner,
        )

    async def save(self, request_id: int, token_ids: list[int]) -> dict[str, Any]:
        """Sync metadata update on master; caller may then ``submit_push``."""
        if self._store is None:
            return {"ok": False, "reason": "no_store"}
        keys = block_keys_from_tokens(token_ids, self.block_size)
        if not keys:
            return {"ok": False, "reason": "empty"}
        return await self._owner.request(
            self._store,
            KVUpdateMsg,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
        )

    def after_alloc_load(
        self,
        request_id: int,
        num_tokens: int,
        local_block_ids: Optional[list[int]] = None,
    ) -> None:
        """After local GPU allocate: start async RDMA/pull TimeoutKernel."""
        _ = local_block_ids  # reserved for future connector meta
        self._submit_transfer(request_id, num_tokens, direction="pull")

    def submit_push(self, request_id: int, num_tokens: int) -> None:
        """After successful ``save``: async push of KV bytes to the pool."""
        self._submit_transfer(request_id, num_tokens, direction="push")

    def _submit_transfer(
        self,
        request_id: int,
        num_tokens: int,
        *,
        direction: TransferDirection,
    ) -> None:
        wid = self._next_workload_id
        self._next_workload_id += 1
        duration_s = self.transfer_duration_s(num_tokens)
        workload = self._workload_generator(
            workload_id=wid,
            request_id=int(request_id),
            duration_s=duration_s,
            direction=direction,
            num_tokens=int(num_tokens),
        )
        self._inflight[wid] = (int(request_id), direction)
        if self._profile is not None:
            start_s = float(self._owner.sim.now())
            self._profile.emit_kv_transfer(
                start_s=start_s,
                duration_s=duration_s,
                replica_id=self._replica_id,
                request_id=int(request_id),
                direction=str(direction),
                num_tokens=int(num_tokens),
            )
        self._engine.send_workload(workload)

    def _handle_complete(self, workload_id: int) -> None:
        entry = self._inflight.pop(int(workload_id), None)
        if entry is None:
            return
        request_id, direction = entry
        self._on_transfer_complete(int(workload_id), int(request_id), direction)
