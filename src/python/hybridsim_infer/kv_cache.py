"""Block-based KV cache manager (prefix match + allocate/free/preempt)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class KvBlock:
    block_id: int
    token_count: int = 0


@dataclass
class KvCacheManager:
    """Simulation-side GPU KV capacity + simple prefix cache.

    ``allocate(request, num_tokens)`` ensures the request has enough blocks to
    hold ``num_computed_tokens + num_tokens`` (vLLM-like slot growth), reusing
    already-allocated blocks when capacity remains.
    """

    num_gpu_blocks: int = 1024
    block_size: int = 16
    free_blocks: int = 1024
    allocated: dict[int, list[KvBlock]] = field(default_factory=dict)
    #: Full prompt token sequences that have been cached locally.
    _prefix_entries: list[list[int]] = field(default_factory=list)
    _next_block_id: int = 0

    def __post_init__(self) -> None:
        # Match vLLM: one block id is reserved as the null block and is not allocatable.
        self._null_reserved = 1 if self.num_gpu_blocks > 0 else 0
        self.free_blocks = max(0, self.num_gpu_blocks - self._null_reserved)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def capacity_tokens(self, request: Any) -> int:
        rid = getattr(request, "request_id", id(request))
        return len(self.allocated.get(rid, [])) * self.block_size

    def match(self, request: Any) -> int:
        """Return longest local prefix-cache hit length (tokens)."""
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        if not tokens:
            return 0
        best = 0
        for cached in self._prefix_entries:
            n = 0
            lim = min(len(cached), len(tokens))
            while n < lim and cached[n] == tokens[n]:
                n += 1
            if n > best:
                best = n
        return best

    def cache_prefix(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        ids = list(token_ids)
        for cached in self._prefix_entries:
            if cached == ids:
                return
        self._prefix_entries.append(ids)

    def blocks_needed_to_hold(self, request: Any, num_tokens: int) -> int:
        """Extra free blocks required so ``request`` can hold ``num_tokens`` total."""
        rid = getattr(request, "request_id", id(request))
        need_blocks = self.blocks_for_tokens(int(num_tokens))
        have = len(self.allocated.get(rid, []))
        return max(0, need_blocks - have)

    def can_fit(self, request: Any, num_tokens: int, *, reserved_blocks: int = 0) -> bool:
        """Whether ``num_tokens`` total capacity fits in free blocks (minus reserved)."""
        grow = self.blocks_needed_to_hold(request, num_tokens)
        return grow <= max(0, self.free_blocks - reserved_blocks)

    def allocate(self, request: Any, num_tokens: int) -> Optional[list[KvBlock]]:
        """Grow blocks so request can hold computed + ``num_tokens``.

        Returns newly allocated blocks (possibly ``[]``), or ``None`` on OOM.
        """
        if num_tokens < 0:
            return None
        rid = getattr(request, "request_id", id(request))
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        need_tokens = computed + int(num_tokens)
        grow = self.blocks_needed_to_hold(request, need_tokens)
        if grow <= 0:
            return []
        if grow > self.free_blocks:
            return None
        blocks: list[KvBlock] = []
        for _ in range(grow):
            bid = self._next_block_id
            self._next_block_id += 1
            blocks.append(KvBlock(block_id=bid, token_count=self.block_size))
        self.free_blocks -= grow
        self.allocated.setdefault(rid, []).extend(blocks)
        return blocks

    def free(self, request: Any) -> None:
        rid = getattr(request, "request_id", id(request))
        blocks = self.allocated.pop(rid, [])
        self.free_blocks += len(blocks)

    def preempt(self, request: Any) -> None:
        """Release all blocks and reset computed tokens (FCFS/vLLM-style)."""
        self.free(request)
        request.num_computed_tokens = 0
        request.pending_remote_tokens = 0
