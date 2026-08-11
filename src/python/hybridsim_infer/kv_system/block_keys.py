"""Block-key helpers aligned with vLLM prefix-cache chaining (pickle SHA-256).

Uses the same chaining as ``vllm.v1.core.kv_cache_utils.hash_block_tokens`` with
``vllm.utils.hashing.sha256`` (pickle + SHA-256). Set ``PYTHONHASHSEED=0`` so
``NONE_HASH`` matches across processes.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Callable, Optional, Sequence

# Parent hash for the first block (mirrors vLLM init_none_hash + sha256).
_NONE_HASH: Optional[bytes] = None


def _sha256_pickle(obj: object) -> bytes:
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def none_hash(hash_fn: Callable[[object], bytes] | None = None) -> bytes:
    """Return NONE_HASH; prefer PYTHONHASHSEED when set."""
    global _NONE_HASH
    fn = hash_fn or _sha256_pickle
    seed = os.getenv("PYTHONHASHSEED")
    if seed is None:
        # Deterministic fallback for sims when seed unset (not identical to
        # vLLM's os.urandom path — always set PYTHONHASHSEED=0 for alignment).
        return fn("0")
    return fn(seed)


def get_none_hash() -> bytes:
    global _NONE_HASH
    if _NONE_HASH is None:
        _NONE_HASH = none_hash()
    return _NONE_HASH


def reset_none_hash() -> None:
    global _NONE_HASH
    _NONE_HASH = None


def hash_block_tokens(
    parent_block_hash: bytes | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple | None = None,
    *,
    hash_fn: Callable[[object], bytes] | None = None,
) -> bytes:
    """Chain-hash one full block (vLLM ``hash_block_tokens`` compatible)."""
    fn = hash_fn or _sha256_pickle
    parent = parent_block_hash if parent_block_hash is not None else get_none_hash()
    return fn((parent, tuple(curr_block_token_ids), extra_keys))


def block_hashes_from_tokens(
    token_ids: Sequence[int],
    block_size: int,
    *,
    hash_fn: Callable[[object], bytes] | None = None,
) -> list[bytes]:
    """Return chained block hashes for each full ``block_size`` chunk."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    ids = list(token_ids)
    n_full = len(ids) // block_size
    out: list[bytes] = []
    parent: bytes | None = None
    for i in range(n_full):
        chunk = ids[i * block_size : (i + 1) * block_size]
        h = hash_block_tokens(parent, chunk, None, hash_fn=hash_fn)
        out.append(h)
        parent = h
    return out


def block_keys_from_tokens(
    token_ids: Sequence[int],
    block_size: int,
) -> list[str]:
    """Hex digests of chained block hashes (Store registry keys in hybridsim)."""
    return [h.hex() for h in block_hashes_from_tokens(token_ids, block_size)]


def block_keys_from_hash_ids(hash_ids: Sequence[int | str]) -> list[str]:
    """Stable Store keys from precomputed prefix-block identities (trace ``hash_ids``).

    Public Mooncake / kvcache-simulator traces publish remapped integer block ids
    in prefix order. Using them directly mirrors vLLM APC lookup on already-known
    block hashes, without re-hashing placeholder tokens.
    """
    keys: list[str] = []
    for hid in hash_ids:
        if isinstance(hid, bool):
            raise ValueError("hash_ids must not contain booleans")
        if isinstance(hid, int):
            keys.append(str(hid))
        else:
            text = str(hid).strip()
            if not text:
                raise ValueError("hash_ids entries must be non-empty")
            keys.append(text)
    return keys


def block_aligned_tokens(num_tokens: int, block_size: int) -> int:
    """Floor to a whole number of blocks (Mooncake-style)."""
    if block_size <= 0 or num_tokens <= 0:
        return 0
    return (num_tokens // block_size) * block_size


def prefix_hit_tokens(
    num_hit_blocks: int,
    input_length: int,
    block_size: int,
) -> int:
    """Token count for a contiguous cached prefix of ``num_hit_blocks`` blocks.

    Full blocks contribute ``block_size`` tokens; a trailing partial block is
    capped by ``input_length`` (kvcache-simulator / Mooncake accounting).
    """
    if num_hit_blocks <= 0 or block_size <= 0:
        return 0
    if input_length <= 0:
        return int(num_hit_blocks) * int(block_size)
    return min(int(input_length), int(num_hit_blocks) * int(block_size))


def full_block_count(num_tokens: int, block_size: int) -> int:
    """Number of complete blocks covered by ``num_tokens`` (vLLM APC style)."""
    if num_tokens <= 0 or block_size <= 0:
        return 0
    return int(num_tokens) // int(block_size)


def resolve_block_keys(
    *,
    token_ids: Sequence[int] | None = None,
    hash_ids: Sequence[int | str] | None = None,
    block_size: int,
    num_tokens: int | None = None,
    input_length: int | None = None,
) -> list[str]:
    """Resolve Store/APC keys: prefer precomputed ``hash_ids``, else vLLM chain hash.

    Truncation (vLLM full-block APC + Mooncake partial last block):
    - ``num_tokens is None`` → all keys / all full token blocks.
    - Else only fully covered blocks (``num_tokens // block_size``).
    - If ``num_tokens >= input_length`` (whole prompt), include the trace's
      partial last ``hash_id`` when present.
    """
    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be positive")

    if hash_ids:
        keys = block_keys_from_hash_ids(hash_ids)
        if num_tokens is None:
            return keys
        n = max(0, int(num_tokens))
        il = int(input_length) if input_length is not None else None
        if il is not None and il > 0 and n >= il:
            return keys
        return keys[: full_block_count(n, bs)]

    tokens = list(token_ids or [])
    if num_tokens is not None:
        tokens = tokens[: max(0, int(num_tokens))]
    return block_keys_from_tokens(tokens, bs)
