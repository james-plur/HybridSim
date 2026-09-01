"""Local GPU KV + optional Store protocol (not transfer duration)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KvStoreConfig:
    """Shared remote Store capacity. Used when ``KvConfig.enable_store``."""

    #: Store object size in tokens. Must be a multiple of GPU ``block_size``.
    #: ``None`` (default) uses the GPU page size (one Store key per GPU page).
    block_size: int | None = None
    #: Remote KV store DRAM capacity in blocks (``<=0`` → unlimited).
    num_blocks: int = 4096


@dataclass
class KvLookupConfig:
    """Store / PD control-plane lookup (not the data-plane α-β model)."""

    #: Fire-and-forget Store lookup + ReplyMsg; pending ≈ vLLM ``None``.
    async_: bool = False
    #: Simulated lookup / Prefill control-plane RTT (seconds).
    rtt_s: float = 1e-3


@dataclass
class KvConfig:
    """GPU page/capacity, local APC, Store switch, and lookup protocol."""

    #: GPU KV page size in tokens.
    block_size: int = 16
    #: Local GPU KV capacity in pages.
    num_gpu_blocks: int = 1024
    #: Local token-list prefix cache (not vLLM APC hashes). Off by default.
    enable_prefix_caching: bool = False
    #: Wire KvClient (per replica) + shared KvStoreActor (monolith and PD).
    enable_store: bool = False
    store: KvStoreConfig = field(default_factory=KvStoreConfig)
    lookup: KvLookupConfig = field(default_factory=KvLookupConfig)

    def resolved_store_block_size(self) -> int:
        """Store page in tokens; equals GPU ``block_size`` when unset."""
        gpu = int(self.block_size)
        raw = self.store.block_size
        if raw is None:
            return gpu
        store = int(raw)
        if store < gpu or store % gpu != 0:
            raise ValueError(
                f"kv.store.block_size ({store}) must be a positive multiple of "
                f"kv.block_size ({gpu})"
            )
        return store
