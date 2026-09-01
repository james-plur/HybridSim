"""In-memory metrics and optional artifact writers for inference runs."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from hybridsim_infer.request import InferenceRequest


def summarize_metrics(
    finished: list[Any],
    *,
    n_scheduled: int,
    sim_now: float,
) -> dict[str, Any]:
    """Aggregate TTFT / TPS / prefix-hit rate from finished requests."""
    if not finished:
        return {
            "mean_ttft_s": None,
            "tps": 0.0,
            "hit_rate": 0.0,
            "n_finished": 0,
            "n_scheduled": int(n_scheduled),
            "sim_now_s": float(sim_now),
            "prefill_tokens": 0,
            "prefix_hit_tokens": 0,
        }
    ttfts: list[float] = []
    for req in finished:
        finished_at = getattr(req, "finished_at", None)
        if finished_at is None:
            continue
        ttfts.append(float(finished_at) - float(req.arrived_at))
    prefill = sum(int(req.num_prefill_tokens) for req in finished)
    hits = sum(int(getattr(req, "prefix_hit_tokens", 0) or 0) for req in finished)
    t0 = min(float(req.arrived_at) for req in finished)
    t1 = max(float(getattr(req, "finished_at", None) or t0) for req in finished)
    span = max(t1 - t0, 1e-12)
    return {
        "mean_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "tps": float(prefill) / span,
        "hit_rate": (float(hits) / float(prefill)) if prefill else 0.0,
        "n_finished": len(finished),
        "n_scheduled": int(n_scheduled),
        "sim_now_s": float(sim_now),
        "prefill_tokens": int(prefill),
        "prefix_hit_tokens": int(hits),
    }


def request_record(req: InferenceRequest) -> dict[str, Any]:
    """Stable per-request row for ``requests.jsonl``."""
    status = getattr(req, "status", None)
    status_name = status.name if hasattr(status, "name") else str(status)
    return {
        "request_id": int(req.request_id),
        "arrived_at": float(req.arrived_at),
        "finished_at": (
            None if req.finished_at is None else float(req.finished_at)
        ),
        "num_prefill_tokens": int(req.num_prefill_tokens),
        "num_decode_tokens": int(req.num_decode_tokens),
        "num_computed_tokens": int(req.num_computed_tokens),
        "num_output_tokens": int(req.num_output_tokens),
        "prefix_hit_tokens": int(getattr(req, "prefix_hit_tokens", 0) or 0),
        "completed": bool(req.completed),
        "status": status_name,
    }


def config_to_dict(config: Any) -> dict[str, Any]:
    """JSON-friendly snapshot; non-serializable injects become type names."""
    converted = _jsonify(config)
    if not isinstance(converted, dict):
        raise TypeError("config_to_dict expects a dataclass instance")
    return converted


def resolve_artifact_path(
    *,
    enabled: bool,
    path: Optional[Path],
    output_dir: Optional[Path],
    default_name: str,
) -> Optional[Path]:
    if not enabled:
        return None
    if path is not None:
        return Path(path)
    if output_dir is not None:
        return Path(output_dir) / default_name
    return Path(default_name)


def _jsonify(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {}
        for item in fields(value):
            out[item.name] = _jsonify(getattr(value, item.name))
        return out
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
