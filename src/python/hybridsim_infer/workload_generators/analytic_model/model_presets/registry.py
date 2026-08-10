"""Load ModelConfig presets generated from elinx/llm-mem-calculator data.js."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hybridsim_infer.workload_generators.analytic_model.configs import ModelConfig

_PRESETS_ROOT = Path(__file__).resolve().parent
_FAMILY_DIRS = ("deepseek", "glm", "kimi", "llama")

# YAML keys that are metadata / not ModelConfig fields.
_META_KEYS = frozenset({"id", "label", "family", "source_url", "dense_intermediate_size"})


def _preset_paths() -> list[Path]:
    paths: list[Path] = []
    for fam in _FAMILY_DIRS:
        d = _PRESETS_ROOT / fam
        if d.is_dir():
            paths.extend(sorted(d.glob("*.yaml")))
    return paths


def list_presets(family: str | None = None) -> list[str]:
    """Return preset ids, optionally filtered by family (e.g. ``deepseek``)."""
    fam = (family or "").lower().strip() or None
    ids: list[str] = []
    for path in _preset_paths():
        if fam is not None and path.parent.name != fam:
            continue
        ids.append(path.stem)
    return ids


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Preset {path} must be a mapping, got {type(data)!r}")
    return data


def _yaml_to_model_config(data: dict[str, Any]) -> ModelConfig:
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _META_KEYS:
            continue
        if hasattr(ModelConfig, key) or key in ModelConfig.__dataclass_fields__:
            kwargs[key] = value
    return ModelConfig(**kwargs)


def load_preset(preset_id: str) -> ModelConfig:
    """Load a preset by id (filename stem, e.g. ``deepseek-v3``)."""
    pid = str(preset_id).strip()
    if not pid:
        raise ValueError("preset_id must be non-empty")
    matches = [p for p in _preset_paths() if p.stem == pid]
    if not matches:
        known = ", ".join(list_presets())
        raise KeyError(f"Unknown model preset {pid!r}. Known: {known}")
    return _yaml_to_model_config(_load_yaml(matches[0]))


def load_model_config(preset_id: str) -> ModelConfig:
    """Alias for :func:`load_preset`."""
    return load_preset(preset_id)


def preset_meta(preset_id: str) -> dict[str, Any]:
    """Return raw YAML metadata for a preset (including non-ModelConfig keys)."""
    pid = str(preset_id).strip()
    matches = [p for p in _preset_paths() if p.stem == pid]
    if not matches:
        raise KeyError(f"Unknown model preset {pid!r}")
    return _load_yaml(matches[0])
