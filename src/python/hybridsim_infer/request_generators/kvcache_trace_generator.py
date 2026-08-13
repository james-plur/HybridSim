"""Load Mooncake / kvcache-simulator JSONL traces into InferenceRequest lists.

Public traces typically provide remapped ``hash_ids`` + lengths (not raw
``input_ids``). This generator:

- maps ``input_length`` / ``output_length`` → prefill/decode lengths
- carries ``hash_ids`` / ``block_size`` on the request for Store prefix reuse
- optionally materializes placeholder ``prompt_token_ids`` of the right length
  so schedulers that still require a token list keep working

For real tokens + hash chain, convert SGLang Finish logs (see
``kvcache-blog/scripts/sglang-log-to-kvcache-trace.py``) and pass
``prompt_token_ids`` via an extended JSONL field ``input_ids`` when present.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO, Union

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators.base import RequestGenerator

PathLike = Union[str, Path]

#: Package-local KV trace tree (raw / normalized / samples / …).
KVCACHE_TRACES_DIR = Path(__file__).resolve().parent / "kvcache_traces"


def _open_text(path: PathLike) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("hash_ids/input_ids must not contain booleans")
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            out.append(int(item))
        else:
            raise ValueError(f"unsupported id value: {item!r}")
    return out


def _placeholder_tokens_from_hash_ids(
    hash_ids: list[int],
    input_length: int,
    block_size: int,
) -> list[int]:
    """Build length-matched placeholder tokens that are unique per block id.

    These are *not* invertible to the original prompt and must not be re-hashed
    with vLLM's chain hash if ``hash_ids`` are already authoritative. They exist
    so code paths that index ``prompt_token_ids[:n]`` keep working.
    """
    if input_length <= 0:
        return []
    if block_size <= 0:
        block_size = max(1, input_length // max(1, len(hash_ids)))
    tokens: list[int] = []
    for index, hid in enumerate(hash_ids):
        remaining = input_length - len(tokens)
        if remaining <= 0:
            break
        n = min(block_size, remaining)
        # Keep values in a tokenizer-ish range; encode block identity in low bits.
        base = (int(hid) % 40_000) + 1
        tokens.extend([(base + j) % 50_000 for j in range(n)])
    if len(tokens) < input_length:
        pad = input_length - len(tokens)
        tokens.extend([(i + 1) % 50_000 for i in range(pad)])
    return tokens[:input_length]


def map_kvcache_trace_record(
    record: dict[str, Any],
    *,
    request_id: int,
    default_block_size: int = 0,
    time_scale: float = 1.0,
    synthesize_prompt_tokens: bool = True,
) -> Optional[InferenceRequest]:
    """Map one Mooncake / Bailian / Weka-flattened JSON object to InferenceRequest."""
    hash_ids = _as_int_list(record.get("hash_ids"))
    input_ids = _as_int_list(record.get("input_ids") or record.get("prompt_token_ids"))
    try:
        input_length = int(record.get("input_length") or record.get("in") or 0)
    except (TypeError, ValueError):
        input_length = 0
    if input_length <= 0 and input_ids:
        input_length = len(input_ids)
    if input_length <= 0:
        return None
    try:
        output_length = int(record.get("output_length") or record.get("out") or 0)
    except (TypeError, ValueError):
        output_length = 0
    try:
        block_size = int(record.get("block_size") or default_block_size or 0)
    except (TypeError, ValueError):
        block_size = default_block_size
    try:
        timestamp = float(record.get("timestamp") or record.get("t") or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0

    if input_ids:
        prompt_token_ids = input_ids[:input_length]
        if len(prompt_token_ids) < input_length:
            prompt_token_ids = prompt_token_ids + [0] * (input_length - len(prompt_token_ids))
    elif synthesize_prompt_tokens and hash_ids:
        prompt_token_ids = _placeholder_tokens_from_hash_ids(
            hash_ids, input_length, block_size
        )
    else:
        prompt_token_ids = []

    return InferenceRequest(
        request_id=int(request_id),
        arrived_at=float(timestamp) * float(time_scale),
        num_prefill_tokens=int(input_length),
        num_decode_tokens=max(0, int(output_length)),
        prompt_token_ids=prompt_token_ids,
        hash_ids=hash_ids,
        block_size=int(block_size) if block_size > 0 else 0,
    )


def iter_kvcache_trace_records(path: PathLike) -> Iterable[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(record, dict):
                continue
            yield record


class KvCacheTraceRequestGenerator(RequestGenerator):
    """RequestGenerator backed by a kvcache-simulator / Mooncake JSONL trace."""

    def __init__(
        self,
        trace_path: PathLike,
        *,
        block_size: int = 0,
        max_requests: Optional[int] = None,
        id_offset: int = 0,
        time_offset: float = 0.0,
        time_scale: float = 1.0,
        synthesize_prompt_tokens: bool = True,
        require_hash_ids: bool = True,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.block_size = int(block_size or 0)
        self.max_requests = max_requests
        self.id_offset = int(id_offset)
        self.time_offset = float(time_offset)
        self.time_scale = float(time_scale)
        self.synthesize_prompt_tokens = bool(synthesize_prompt_tokens)
        self.require_hash_ids = bool(require_hash_ids)

    def generate(self) -> list[InferenceRequest]:
        if not self.trace_path.exists():
            raise FileNotFoundError(self.trace_path)
        requests: list[InferenceRequest] = []
        for record in iter_kvcache_trace_records(self.trace_path):
            if self.require_hash_ids and not record.get("hash_ids"):
                continue
            req = map_kvcache_trace_record(
                record,
                request_id=self.id_offset + len(requests),
                default_block_size=self.block_size,
                time_scale=self.time_scale,
                synthesize_prompt_tokens=self.synthesize_prompt_tokens,
            )
            if req is None:
                continue
            req.arrived_at = float(req.arrived_at) + self.time_offset
            requests.append(req)
            if self.max_requests is not None and len(requests) >= int(self.max_requests):
                break
        requests.sort(key=lambda r: (r.arrived_at, r.request_id))
        # Re-assign dense ids after sort for stable scheduling demos.
        for index, req in enumerate(requests):
            req.request_id = self.id_offset + index
        return requests
