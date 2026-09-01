"""Compute-duration workload: batch-level predictors or op-level DAG."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybridsim_infer.workload_generators.configs import (
    DeviceConfig,
    ModelConfig,
    NetworkConfig,
    OpLevelConfig,
    ParallelConfig,
)

_VALID_MODES = frozenset({"batch_level", "op_level"})
_VALID_PREDICTORS = frozenset({"fixed", "token_proportional", "frontier"})


@dataclass
class BatchFixedConfig:
    #: Dummy TimeoutKernel duration when ``predictor=fixed``.
    dummy_exec_s: float = 0.05


@dataclass
class BatchTokenProportionalConfig:
    prefill_s_per_token: float = 1e-4
    decode_s_per_token: float = 1e-3
    base_s: float = 0.0


@dataclass
class BatchFrontierConfig:
    #: Injected Frontier predictor (not serialized).
    predictor: object | None = None
    #: Frontier ``ClusterType`` (default MONOLITHIC). Not ``cluster.type``.
    cluster_type: object | None = None
    replica_id: int = 0
    is_moe: bool = False


@dataclass
class BatchLevelConfig:
    #: ``fixed`` / ``token_proportional`` / ``frontier`` (batch_level only).
    predictor: str = "fixed"
    fixed: BatchFixedConfig = field(default_factory=BatchFixedConfig)
    token_proportional: BatchTokenProportionalConfig = field(
        default_factory=BatchTokenProportionalConfig
    )
    frontier: BatchFrontierConfig = field(default_factory=BatchFrontierConfig)


@dataclass
class InferWorkloadConfig:
    """How a scheduled batch becomes Engine TimeoutKernel duration."""

    #: ``batch_level`` (predictor) or ``op_level`` (mock DAG + Roofline / α-β).
    mode: str = "batch_level"
    batch: BatchLevelConfig = field(default_factory=BatchLevelConfig)
    #: Op-level DAG / Roofline / collective network (also carries ModelConfig
    #: for KV volume when ``model.preset`` is unset).
    op: OpLevelConfig = field(default_factory=OpLevelConfig)

    def resolved_mode(self) -> str:
        mode = (self.mode or "").lower().strip()
        if mode not in _VALID_MODES:
            raise ValueError(
                "infer_workload.mode must be 'batch_level' or 'op_level', "
                f"got {self.mode!r}"
            )
        return mode

    def resolved_predictor(self) -> str:
        pred = (self.batch.predictor or "").lower().strip()
        if pred not in _VALID_PREDICTORS:
            raise ValueError(
                "infer_workload.batch.predictor must be 'fixed', "
                f"'token_proportional', or 'frontier', got {self.batch.predictor!r}"
            )
        return pred

    def apply_model(self, model: ModelConfig | None = None) -> OpLevelConfig:
        """Write a resolved ``ModelConfig`` onto ``op`` when provided."""
        if model is not None:
            self.op.model = model
        return self.op


__all__ = [
    "BatchFixedConfig",
    "BatchFrontierConfig",
    "BatchLevelConfig",
    "BatchTokenProportionalConfig",
    "DeviceConfig",
    "InferWorkloadConfig",
    "ModelConfig",
    "NetworkConfig",
    "OpLevelConfig",
    "ParallelConfig",
]
