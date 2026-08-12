"""Factories for workload generators."""

from __future__ import annotations

from typing import Any, Optional

from hybridsim_infer.workload_generators.base import WorkloadGenerator
from hybridsim_infer.workload_generators.model_config_resolve import (
    resolve_analytical_config,
)
from hybridsim_infer.workload_generators.predict_workload_generator import (
    PredictWorkloadGenerator,
)
from hybridsim_infer.workload_generators.predictors import (
    BatchDurationPredictor,
    make_predictor,
)


def make_workload_generator(
    *,
    duration_mode: str = "fixed",
    dummy_exec_s: float = 0.05,
    prefill_s_per_token: float = 1e-4,
    decode_s_per_token: float = 1e-3,
    duration_base_s: float = 0.0,
    predictor: Optional[BatchDurationPredictor] = None,
    frontier_predictor: Any = None,
    frontier_cluster_type: Any = None,
    frontier_replica_id: int = 0,
    frontier_is_moe: bool = False,
    analytical_config: Any = None,
    model_preset: Optional[str] = None,
    model_config: Any = None,
    parallel_config: Any = None,
    device_config: Any = None,
    network_config: Any = None,
) -> WorkloadGenerator:
    """Build a workload generator.

    ``duration_mode``:
      - ``fixed`` / ``token_proportional`` → ``PredictWorkloadGenerator`` + built-ins
      - ``predict`` → Frontier RF wrapper (requires Frontier + ``frontier_predictor``)
      - ``analytical`` → ``OpWorkloadGenerator`` (Roofline / α-β Operator DAG)

    ``model_preset`` (when set) injects a shared ``ModelConfig`` into
    ``analytical_config`` for every mode (KV shape + analytical DAG).
    """
    analytical_config = resolve_analytical_config(
        analytical_config=analytical_config,
        model_preset=model_preset,
    )

    mode = (duration_mode or "fixed").lower().strip()
    if mode == "predict":
        return _make_frontier_predict_workload_generator(
            predictor=predictor,
            frontier_predictor=frontier_predictor,
            frontier_cluster_type=frontier_cluster_type,
            frontier_replica_id=frontier_replica_id,
            frontier_is_moe=frontier_is_moe,
        )
    if mode in ("analytical", "analytic", "op", "kernel_dag"):
        return _make_analytical_workload_generator(
            analytical_config=analytical_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device_config=device_config,
            network_config=network_config,
        )

    pred = predictor or make_predictor(
        duration_mode=duration_mode,
        dummy_exec_s=dummy_exec_s,
        prefill_s_per_token=prefill_s_per_token,
        decode_s_per_token=decode_s_per_token,
        base_s=duration_base_s,
    )
    return PredictWorkloadGenerator(pred)


def _make_analytical_workload_generator(
    *,
    analytical_config: Any,
    model_config: Any,
    parallel_config: Any,
    device_config: Any,
    network_config: Any,
) -> WorkloadGenerator:
    from hybridsim_infer.workload_generators.analytic_model.configs import (
        AnalyticalConfig,
    )
    from hybridsim_infer.workload_generators.op_workload_generator import (
        OpWorkloadGenerator,
    )

    if isinstance(analytical_config, AnalyticalConfig):
        return OpWorkloadGenerator(analytical=analytical_config)
    if analytical_config is not None:
        raise TypeError(
            "analytical_config must be AnalyticalConfig or None, "
            f"got {type(analytical_config)!r}"
        )
    return OpWorkloadGenerator(
        model=model_config,
        parallel=parallel_config,
        device=device_config,
        network=network_config,
    )


def _make_frontier_predict_workload_generator(
    *,
    predictor: Optional[BatchDurationPredictor],
    frontier_predictor: Any,
    frontier_cluster_type: Any,
    frontier_replica_id: int,
    frontier_is_moe: bool,
) -> WorkloadGenerator:
    try:
        from hybridsim_infer.workload_generators.predictors.frontier import (
            FrontierBatchDurationPredictor,
        )
        from frontier.types import ClusterType
    except ImportError as exc:
        raise ImportError(
            "duration_mode='predict' requires the Frontier package "
            "(PYTHONPATH to Frontier or pip install -e $FRONTIER_ROOT)"
        ) from exc

    if isinstance(predictor, FrontierBatchDurationPredictor):
        return PredictWorkloadGenerator(predictor)

    if frontier_predictor is None:
        raise ValueError(
            "duration_mode='predict' requires frontier_predictor="
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
    return PredictWorkloadGenerator(wrap)
