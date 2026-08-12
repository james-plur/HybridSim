"""Resolve ``model_preset`` / ``AnalyticalConfig`` for all workload paths."""

from __future__ import annotations

from typing import Any, Optional


def resolve_analytical_config(
    *,
    analytical_config: Any = None,
    model_preset: Optional[str] = None,
) -> Any:
    """Inject ``model_preset`` into ``AnalyticalConfig`` when requested.

    Shared by builder / factory so every ``duration_mode`` can hang the same
    ``ModelConfig`` (DAG shape, KV bytes). Predict-mode RF timing still comes
    from ``frontier_predictor``; preset and RF should describe the same model family.
    """
    if not model_preset:
        return analytical_config

    from hybridsim_infer.workload_generators.analytic_model.configs import (
        AnalyticalConfig,
    )
    from hybridsim_infer.workload_generators.analytic_model.model_presets import (
        load_preset,
    )

    model = load_preset(str(model_preset))
    if analytical_config is None:
        return AnalyticalConfig(model=model)
    if hasattr(analytical_config, "model"):
        analytical_config.model = model
        return analytical_config
    return AnalyticalConfig(model=model)


def resolve_model_config(
    *,
    analytical_config: Any = None,
    model_preset: Optional[str] = None,
    model_config: Any = None,
) -> Any:
    """Return the effective ``ModelConfig`` from preset / analytical / explicit."""
    resolved = resolve_analytical_config(
        analytical_config=analytical_config,
        model_preset=model_preset,
    )
    if resolved is not None and hasattr(resolved, "model"):
        return resolved.model
    return model_config
