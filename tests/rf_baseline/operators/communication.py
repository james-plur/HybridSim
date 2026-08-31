"""Hidden-state payload helper for frozen collective costing."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)


def hidden_state_payload_bytes(
    *,
    model: ModelConfig,
    batch: BatchFeatures,
) -> int:
    return int(batch.num_tokens) * int(model.hidden_size) * int(model.dtype_bytes)
