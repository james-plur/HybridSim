"""Shared model identity for infer DAG shape and KV transfer volume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    """Model preset and/or explicit ``ModelConfig``.

    Shared across all ``infer_workload.mode`` values: injects ``ModelConfig``
    into the op-level DAG and KV transfer volume. Explicit ``config`` overrides
    a loaded preset. For ``batch.predictor=frontier``, RF timing still comes
    from ``frontier.predictor``; preset and RF should describe the same family.
    """

    #: Preset id (e.g. ``llama-3.1-8b`` / ``deepseek-v3``).
    preset: str | None = None
    #: Explicit ``ModelConfig``. When set, wins over ``preset``.
    config: Any = None

    def resolve(self) -> Any:
        """Return the effective ``ModelConfig``, or ``None`` if unspecified."""
        if self.config is not None:
            return self.config
        if not self.preset:
            return None
        from hybridsim_infer.workload_generators.model_presets import load_preset

        return load_preset(str(self.preset))
