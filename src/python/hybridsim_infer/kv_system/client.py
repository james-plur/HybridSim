"""KvClient: replica-local Mooncake-style store client + transfer engine."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from hybridsim_infer.kv_system.block_keys import (
    block_aligned_tokens,
    prefix_hit_tokens,
    resolve_block_keys,
)
from hybridsim_infer.messages import KVLookupMsg, KVLookupReplyMsg, KVUpdateMsg
from hybridsim_infer.workload_generators.analytic_model.configs import (
    ModelConfig,
    NetworkConfig,
)
from hybridsim_infer.workload_generators.analytic_model.kv_cache import (
    bytes_per_token as model_bytes_per_token,
)
from hybridsim_infer.workload_generators.kv_transfer import (
    KvTransferWorkloadGenerator,
    transfer_duration_s,
)

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
        kv_latency_s: float = 0.0,
        lookup_rtt_s: float = 1e-3,
        on_transfer_complete: Callable[[int, int, str], None],
        workload_generator: Optional[KvTransferWorkloadGenerator] = None,
        model_config: Optional[ModelConfig] = None,
        network_config: Optional[NetworkConfig] = None,
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
        self.kv_latency_s = float(kv_latency_s)
        self.lookup_rtt_s = float(lookup_rtt_s)
        self.model_config = model_config
        self.network_config = network_config
        self._on_transfer_complete = on_transfer_complete
        if workload_generator is not None:
            self._workload_generator = workload_generator
        else:
            net = network_config
            if net is None and (kv_latency_s > 0 or bandwidth_gbps > 0):
                net = NetworkConfig.from_bandwidth(
                    latency_s=kv_latency_s,
                    bandwidth_gbps=bandwidth_gbps,
                )
            self._workload_generator = KvTransferWorkloadGenerator(
                model=model_config,
                network=net,
                bytes_per_token=bytes_per_token,
                bandwidth_gbps=bandwidth_gbps,
                latency_s=kv_latency_s,
                transfer_s_floor=transfer_s_floor,
            )
        # Prefer model-derived bytes/token when available (scalar fallback kept).
        if model_config is not None:
            self.bytes_per_token = float(model_bytes_per_token(model_config, num_tokens=1))
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
        """α-β duration from model KV volume when set, else scalar bytes/token."""
        gen = self._workload_generator
        if hasattr(gen, "estimate_duration_s"):
            return float(gen.estimate_duration_s(int(num_tokens)))
        return transfer_duration_s(
            num_tokens=int(num_tokens),
            model=self.model_config,
            bytes_per_token_fallback=self.bytes_per_token,
            network=self.network_config,
            bandwidth_gbps=self.bandwidth_gbps,
            latency_s=self.kv_latency_s,
            transfer_s_floor=self.transfer_s_floor,
        )

    def control_plane_hit(
        self,
        token_ids: list[int],
        *,
        location: Any,
        num_tokens: int | None = None,
        block_size: int = 0,
    ) -> dict[str, Any]:
        """Full prompt as hit without Store hash match (Prefill GPU known)."""
        bs = int(block_size) if int(block_size) > 0 else self.block_size
        n = int(num_tokens) if num_tokens is not None else len(token_ids)
        aligned = block_aligned_tokens(n, bs)
        return {
            "hit": aligned > 0,
            "num_tokens": aligned,
            "num_blocks": aligned // bs if bs else 0,
            "location": location,
            "mode": "control_plane",
        }

    def lookup_control_plane(
        self,
        request_id: int,
        token_ids: list[int],
        *,
        location: Any,
        num_tokens: int | None = None,
        block_size: int = 0,
    ) -> dict[str, Any]:
        """Fire delayed control-plane reply (RTT); caller treats as pending."""
        result = self.control_plane_hit(
            token_ids,
            location=location,
            num_tokens=num_tokens,
            block_size=block_size,
        )
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

    def _effective_block_size(self, block_size: int = 0) -> int:
        return int(block_size) if int(block_size) > 0 else self.block_size

    def keys_for_prompt(
        self,
        *,
        token_ids: list[int] | None = None,
        hash_ids: list[int] | None = None,
        block_size: int = 0,
        num_tokens: int | None = None,
        input_length: int | None = None,
    ) -> tuple[list[str], int, int]:
        """Return ``(block_keys, tokens_per_block, input_length)`` for Store RPC."""
        bs = self._effective_block_size(block_size)
        tokens = list(token_ids or [])
        hashes = list(hash_ids or [])
        il = int(input_length) if input_length is not None else (
            int(num_tokens) if num_tokens is not None else len(tokens)
        )
        if il <= 0 and tokens:
            il = len(tokens)
        n = int(num_tokens) if num_tokens is not None else il
        keys = resolve_block_keys(
            token_ids=tokens or None,
            hash_ids=hashes or None,
            block_size=bs,
            num_tokens=n if n > 0 else None,
            input_length=il if il > 0 else None,
        )
        return keys, bs, il

    async def lookup(
        self,
        request_id: int,
        token_ids: list[int],
        *,
        hash_ids: list[int] | None = None,
        block_size: int = 0,
        num_tokens: int | None = None,
        input_length: int | None = None,
    ) -> dict[str, Any]:
        """Sync metadata lookup on master (block-aligned hit length)."""
        if self._store is None:
            return {"hit": False, "num_tokens": 0, "num_blocks": 0}
        keys, tpb, il = self.keys_for_prompt(
            token_ids=token_ids,
            hash_ids=hash_ids,
            block_size=block_size,
            num_tokens=num_tokens,
            input_length=input_length,
        )
        return await self._owner.request(
            self._store,
            KVLookupMsg,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
            async_reply=False,
            reply_to=None,
            tokens_per_block=tpb,
            input_length=il,
        )

    def lookup_async(
        self,
        request_id: int,
        token_ids: list[int],
        *,
        hash_ids: list[int] | None = None,
        block_size: int = 0,
        num_tokens: int | None = None,
        input_length: int | None = None,
    ) -> None:
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
        keys, tpb, il = self.keys_for_prompt(
            token_ids=token_ids,
            hash_ids=hash_ids,
            block_size=block_size,
            num_tokens=num_tokens,
            input_length=input_length,
        )
        self._store.send(
            KVLookupMsg,
            delay=self.lookup_rtt_s,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
            async_reply=True,
            reply_to=self._owner,
            tokens_per_block=tpb,
            input_length=il,
        )

    async def save(
        self,
        request_id: int,
        token_ids: list[int],
        *,
        hash_ids: list[int] | None = None,
        block_size: int = 0,
        num_tokens: int | None = None,
        input_length: int | None = None,
    ) -> dict[str, Any]:
        """Sync metadata update on master; caller may then ``submit_push``."""
        if self._store is None:
            return {"ok": False, "reason": "no_store"}
        keys, tpb, il = self.keys_for_prompt(
            token_ids=token_ids,
            hash_ids=hash_ids,
            block_size=block_size,
            num_tokens=num_tokens if num_tokens is not None else len(token_ids),
            input_length=input_length,
        )
        if not keys:
            return {"ok": False, "reason": "empty"}
        reply = await self._owner.request(
            self._store,
            KVUpdateMsg,
            token_ids=list(token_ids),
            block_keys=keys,
            request_id=int(request_id),
            tokens_per_block=tpb,
        )
        # Prefer token accounting from the keys we inserted when using trace blocks.
        if reply and reply.get("ok") and tpb > 0:
            n_blocks = int(reply.get("num_blocks", len(keys)) or len(keys))
            reply = dict(reply)
            reply["num_tokens"] = prefix_hit_tokens(n_blocks, il or n_blocks * tpb, tpb)
            reply["num_blocks"] = n_blocks
        return reply

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
