"""Factories for infer and KV workload generators."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.workload_generators.infer_workload_generator.base import (
    InferWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.generator import (
    BatchLevelWorkloadGenerator,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors import (
    BatchDurationPredictor,
    make_predictor,
)
from hybridsim_infer.workload_generators.kv_workload_generator.generator import (
    KvWorkloadGenerator,
)
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_op_level_config,
)


def make_infer_workload_generator(
    *,
    duration_mode: str = "batch_level",
    batch_predictor: str = "fixed",
    dummy_exec_s: float = 0.05,
    prefill_s_per_token: float = 1e-4,
    decode_s_per_token: float = 1e-3,
    duration_base_s: float = 0.0,
    predictor: Optional[BatchDurationPredictor] = None,
    frontier_predictor: Any = None,
    frontier_cluster_type: Any = None,
    frontier_replica_id: int = 0,
    frontier_is_moe: bool = False,
    op_level_config: Any = None,
    model_preset: Optional[str] = None,
    model_config: Any = None,
    parallel_config: Any = None,
    device_config: Any = None,
    network_config: Any = None,
) -> InferWorkloadGenerator:
    """Build an infer workload generator.

    ``duration_mode``:
      - ``batch_level`` → ``BatchLevelWorkloadGenerator`` + ``batch_predictor``
      - ``op_level`` → ``OpLevelWorkloadGenerator`` (mock DAG + Roofline / α-β)

    ``batch_predictor`` (batch_level only): ``fixed`` / ``token_proportional`` / ``frontier``.
    ``model_preset`` injects a shared ``ModelConfig`` into ``op_level_config``.
    """
    op_level_config = resolve_op_level_config(
        op_level_config=op_level_config,
        model_preset=model_preset,
    )

    mode = (duration_mode or "batch_level").lower().strip()
    if mode == "op_level":
        return _make_op_level_workload_generator(
            op_level_config=op_level_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device_config=device_config,
            network_config=network_config,
        )
    if mode != "batch_level":
        raise ValueError(
            "duration_mode must be 'batch_level' or 'op_level', "
            f"got {duration_mode!r}"
        )

    pred_name = (batch_predictor or "fixed").lower().strip()
    if pred_name == "frontier":
        return _make_frontier_batch_workload_generator(
            predictor=predictor,
            frontier_predictor=frontier_predictor,
            frontier_cluster_type=frontier_cluster_type,
            frontier_replica_id=frontier_replica_id,
            frontier_is_moe=frontier_is_moe,
        )
    pred = predictor or make_predictor(
        batch_predictor=pred_name,
        dummy_exec_s=dummy_exec_s,
        prefill_s_per_token=prefill_s_per_token,
        decode_s_per_token=decode_s_per_token,
        base_s=duration_base_s,
    )
    return BatchLevelWorkloadGenerator(pred)


def make_kv_workload_generator(
    *,
    model: Any = None,
    network: Any = None,
    bytes_per_token: float = 16.0,
    bandwidth_gbps: float = 50.0,
    latency_s: float = 0.0,
    transfer_s_floor: float = 0.0,
    page_tokens: int = 0,
    model_preset: Optional[str] = None,
) -> KvWorkloadGenerator:
    """Build a KV transfer workload generator (shared ``model_preset`` / ModelConfig)."""
    if model is None and model_preset:
        from hybridsim_infer.workload_generators.model_presets import load_preset

        model = load_preset(model_preset)
    return KvWorkloadGenerator(
        model=model,
        network=network,
        bytes_per_token=bytes_per_token,
        bandwidth_gbps=bandwidth_gbps,
        latency_s=latency_s,
        transfer_s_floor=transfer_s_floor,
        page_tokens=page_tokens,
    )


def _make_op_level_workload_generator(
    *,
    op_level_config: Any,
    model_config: Any,
    parallel_config: Any,
    device_config: Any,
    network_config: Any,
) -> InferWorkloadGenerator:
    from hybridsim_infer.workload_generators.configs import OpLevelConfig
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.generator import (
        OpLevelWorkloadGenerator,
    )

    if isinstance(op_level_config, OpLevelConfig):
        return OpLevelWorkloadGenerator(op_level=op_level_config)
    if op_level_config is not None:
        raise TypeError(
            "op_level_config must be OpLevelConfig or None, "
            f"got {type(op_level_config)!r}"
        )
    return OpLevelWorkloadGenerator(
        model=model_config,
        parallel=parallel_config,
        device=device_config,
        network=network_config,
    )


def _make_frontier_batch_workload_generator(
    *,
    predictor: Optional[BatchDurationPredictor],
    frontier_predictor: Any,
    frontier_cluster_type: Any,
    frontier_replica_id: int,
    frontier_is_moe: bool,
) -> InferWorkloadGenerator:
    try:
        from hybridsim_infer.workload_generators.infer_workload_generator.batch_level.predictors.frontier import (
            FrontierBatchDurationPredictor,
        )
        from frontier.types import ClusterType
    except ImportError as exc:
        raise ImportError(
            "batch_predictor='frontier' requires the Frontier package "
            "(PYTHONPATH to Frontier or pip install -e $FRONTIER_ROOT)"
        ) from exc

    if isinstance(predictor, FrontierBatchDurationPredictor):
        return BatchLevelWorkloadGenerator(predictor)

    if frontier_predictor is None:
        raise ValueError(
            "batch_predictor='frontier' requires frontier_predictor="
            "BaseExecutionTimePredictor (e.g. RandomForest from "
            "ExecutionTimePredictorRegistry) or a FrontierBatchDurationPredictor"
        )

    cluster_type = frontier_cluster_type
    if cluster_type is None:
        cluster_type = ClusterType.MONOLITHIC
    wrap = FrontierBatchDurationPredictor(
        frontier_predictor,
        cluster_type=cluster_type,
        replica_id=frontier_replica_id,
        is_moe=frontier_is_moe,
    )
    return BatchLevelWorkloadGenerator(wrap)
