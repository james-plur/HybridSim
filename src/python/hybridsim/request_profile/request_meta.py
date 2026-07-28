"""Serialize InferenceRequest (duck-typed) fields for Chrome Trace metadata."""

from __future__ import annotations

from typing import Any, Optional


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def snapshot_request_meta(
    req: Any,
    *,
    prompt_prefix_len: int = 8,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a JSON-safe request metadata dict for profile storage / event args."""
    prompt = list(getattr(req, "prompt_token_ids", None) or [])
    kv = getattr(req, "kv_transfer_params", None) or {}
    status = getattr(req, "status", None)
    meta: dict[str, Any] = {
        "request_id": int(req.request_id),
        "arrived_at": float(getattr(req, "arrived_at", 0.0) or 0.0),
        "num_prefill_tokens": int(getattr(req, "num_prefill_tokens", 0) or 0),
        "num_decode_tokens": int(getattr(req, "num_decode_tokens", 0) or 0),
        "num_computed_tokens": int(getattr(req, "num_computed_tokens", 0) or 0),
        "num_output_tokens": int(getattr(req, "num_output_tokens", 0) or 0),
        "prompt_len": len(prompt),
        "prompt_prefix": [int(x) for x in prompt[: max(0, int(prompt_prefix_len))]],
        "completed": bool(getattr(req, "completed", False)),
        "status": getattr(status, "name", str(status)) if status is not None else None,
        "kv_transfer_params": _json_safe(dict(kv)),
    }
    if extra:
        meta.update(_json_safe(extra))
    return meta
