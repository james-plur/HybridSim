"""Case fixture loading for offline schedule alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CASES_DIR = Path(__file__).resolve().parent / "cases"


@dataclass
class CaseRequest:
    request_id: str
    arrive_step: int
    num_prefill_tokens: int
    num_decode_tokens: int
    prompt_token_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseRequest:
        rid = str(data["request_id"])
        n_prefill = int(data["num_prefill_tokens"])
        prompt = list(data.get("prompt_token_ids") or [])
        if not prompt and n_prefill > 0:
            # Stable synthetic prompt (shared across drivers).
            base = int(data.get("prompt_base", abs(hash(rid)) % 1000))
            prompt = [base + i for i in range(n_prefill)]
        return cls(
            request_id=rid,
            arrive_step=int(data.get("arrive_step", 0)),
            num_prefill_tokens=n_prefill,
            num_decode_tokens=int(data["num_decode_tokens"]),
            prompt_token_ids=prompt,
        )


@dataclass
class CaseSpec:
    name: str
    description: str = ""
    framework: str = "vllm"
    scheduler: dict[str, Any] = field(default_factory=dict)
    requests: list[CaseRequest] = field(default_factory=list)
    seed_prefix_cache: list[list[int]] = field(default_factory=list)
    max_steps: int = 10_000

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, name: str | None = None) -> CaseSpec:
        scheduler = dict(data.get("scheduler") or {})
        framework = str(
            data.get("framework") or scheduler.get("framework") or "vllm"
        )
        return cls(
            name=name or str(data.get("name", "unnamed")),
            description=str(data.get("description", "")),
            framework=framework,
            scheduler=scheduler,
            requests=[CaseRequest.from_dict(r) for r in (data.get("requests") or [])],
            seed_prefix_cache=[
                list(p) for p in (data.get("seed_prefix_cache") or [])
            ],
            max_steps=int(data.get("max_steps", 10_000)),
        )


def load_case(name_or_path: str | Path) -> CaseSpec:
    path = Path(name_or_path)
    if not path.exists():
        path = CASES_DIR / f"{name_or_path}.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return CaseSpec.from_dict(data, name=path.stem)


def list_cases() -> list[str]:
    if not CASES_DIR.exists():
        return []
    return sorted(p.stem for p in CASES_DIR.glob("*.json"))
