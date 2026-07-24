"""Stub KV cache manager (block-based placeholder; extend toward vLLM later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class KvBlock:
    block_id: int
    token_count: int = 0


@dataclass
class KvCacheManager:
    """Simulation-side KV capacity tracker.

    Real prefix match / eviction can replace these stubs later.
    """

    num_gpu_blocks: int = 1024
    block_size: int = 16
    free_blocks: int = 1024
    allocated: dict[int, list[KvBlock]] = field(default_factory=dict)
    _next_block_id: int = 0

    def __post_init__(self) -> None:
        self.free_blocks = self.num_gpu_blocks

    def match(self, request: Any) -> int:
        """Return number of prefix-cached tokens (stub: always 0)."""
        # TODO: align with vLLM prefix cache match
        return 0

    def allocate(self, request: Any, num_tokens: int) -> Optional[list[KvBlock]]:
        """Allocate blocks for ``num_tokens``. Return None on failure."""
        need = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        if need > self.free_blocks:
            return None
        blocks: list[KvBlock] = []
        for _ in range(need):
            bid = self._next_block_id
            self._next_block_id += 1
            blocks.append(KvBlock(block_id=bid, token_count=self.block_size))
        self.free_blocks -= need
        rid = getattr(request, "request_id", id(request))
        self.allocated.setdefault(rid, []).extend(blocks)
        return blocks

    def free(self, request: Any) -> None:
        rid = getattr(request, "request_id", id(request))
        blocks = self.allocated.pop(rid, [])
        self.free_blocks += len(blocks)
