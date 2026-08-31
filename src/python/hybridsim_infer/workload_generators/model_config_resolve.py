"""Resolve ``model_preset`` / ``OpLevelConfig`` for all workload paths."""

from __future__ import annotations

from typing import Any, Optional


def resolve_op_level_config(
    *,
    op_level_config: Any = None,
    model_preset: Optional[str] = None,
) -> Any:
    """Inject ``model_preset`` into ``OpLevelConfig`` when requested.

    Shared by builder / factory so every ``duration_mode`` can hang the same
    ``ModelConfig`` (DAG shape, KV bytes). Batch-level Frontier RF timing still
    comes from ``frontier_predictor``; preset and RF should describe the same
    model family.
    """
    if not model_preset:
        return op_level_config

    from hybridsim_infer.workload_generators.configs import OpLevelConfig
    from hybridsim_infer.workload_generators.model_presets import load_preset

    model = load_preset(str(model_preset))
    if op_level_config is None:
        return OpLevelConfig(model=model)
    if hasattr(op_level_config, "model"):
        op_level_config.model = model
        return op_level_config
    return OpLevelConfig(model=model)


def resolve_model_config(
    *,
    op_level_config: Any = None,
    model_preset: Optional[str] = None,
    model_config: Any = None,
) -> Any:
    """Return the effective ``ModelConfig`` from preset / op-level / explicit."""
    resolved = resolve_op_level_config(
        op_level_config=op_level_config,
        model_preset=model_preset,
    )
    if resolved is not None and hasattr(resolved, "model"):
        return resolved.model
    return model_config
