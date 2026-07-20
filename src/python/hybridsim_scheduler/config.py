"""Hybridsim simulation configuration dataclasses.

Hierarchy (extend with new subclasses for future simulation kinds)::

    SimulationConfig          # shared runtime options
    ├── ArchitectureConfig    # Frontier architecture CLI / case runs
    └── MonolithicConfig      # single-cluster MONOLITHIC smoke / demos

``Simulation`` takes any ``SimulationConfig`` specialization and obtains the
underlying Frontier config via ``to_frontier()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from frontier.config import SimulationConfig as FrontierSimulationConfig

if TYPE_CHECKING:
    from hybridsim_scheduler.frontier_bridge.context import ReplicaSchedulerKind


def load_frontier_config_from_cli_args(cli_args: list[str]) -> FrontierSimulationConfig:
    """Parse Frontier ``SimulationConfig`` from CLI-style argument list."""
    import os
    import sys

    old_argv = sys.argv
    previous_cwd = os.getcwd()
    frontier_root = Path("/home/y_luchenda/Frontier")
    try:
        if frontier_root.exists():
            os.chdir(frontier_root)
        sys.argv = ["hybridsim"] + list(cli_args)
        return FrontierSimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
        os.chdir(previous_cwd)


@dataclass
class SimulationConfig:
    """Base hybridsim simulation configuration.

    Holds options that apply to every simulation kind. Scenario-specific fields
    belong on subclasses (``ArchitectureConfig``, ``MonolithicConfig``, ...).
    Subclasses must implement ``to_frontier()`` so ``Simulation`` can build the
    Frontier scheduler bundle without knowing the concrete config type.
    """

    #: Directory containing the built ``hybridsim_py`` extension module.
    build_dir: Optional[Path] = None
    #: If set, schedule traces are written under this directory.
    trace_output_dir: Optional[Path] = None

    def to_frontier(self) -> FrontierSimulationConfig:
        """Materialize the Frontier ``SimulationConfig`` used by the bridge."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_frontier()"
        )


@dataclass
class ArchitectureConfig(SimulationConfig):
    """Config for Frontier architecture cases (co-location / PDD, online/offline).

    Typically produced by ``load_frontier_config_from_cli_args`` (or
    ``architecture_cases.build_cli_args``) and passed to ``Simulation``.
    """

    #: Pre-built Frontier config; required before constructing ``Simulation``.
    frontier: FrontierSimulationConfig | None = None

    def to_frontier(self) -> FrontierSimulationConfig:
        if self.frontier is None:
            raise ValueError("ArchitectureConfig.frontier is required")
        return self.frontier


@dataclass
class MonolithicConfig(SimulationConfig):
    """Config for a single MONOLITHIC cluster smoke / demo run.

    Use ``Simulation(MonolithicConfig(...))`` as the entry point.
    """

    replica_scheduler_kind: Optional["ReplicaSchedulerKind"] = None
    num_replicas: int = 1
    attn_data_parallel_size: int = 1
    dummy_execution_time_ms: float = 100.0
    #: Alias for metrics output; copied into ``trace_output_dir`` when unset.
    metrics_output_dir: Optional[Path] = None
    run_id: str = "monolithic_smoke"

    def __post_init__(self) -> None:
        if self.trace_output_dir is None and self.metrics_output_dir is not None:
            self.trace_output_dir = self.metrics_output_dir

    def build_cli_args(self) -> list[str]:
        """CLI args that produce an equivalent Frontier MONOLITHIC config."""
        from hybridsim_scheduler.frontier_bridge.context import ReplicaSchedulerKind

        kind = self.replica_scheduler_kind or ReplicaSchedulerKind.VLLM_V1
        metrics_dir = self.metrics_output_dir or self.trace_output_dir
        return [
            "--simulation_mode",
            "offline",
            "--sys_arch",
            "co-location",
            "--cluster_config_num_replicas",
            str(self.num_replicas),
            "--replica_config_attn_data_parallel_size",
            str(self.attn_data_parallel_size),
            "--replica_scheduler_config_type",
            kind.value,
            "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
            str(self.dummy_execution_time_ms),
            "--request_generator_config_type",
            "synthetic",
            "--synthetic_request_generator_config_num_requests",
            "1",
            "--length_generator_config_type",
            "fixed",
            "--fixed_request_length_generator_config_prefill_tokens",
            "4",
            "--fixed_request_length_generator_config_decode_tokens",
            "2",
            "--interval_generator_config_type",
            "poisson",
            "--poisson_request_interval_generator_config_qps",
            "1.0",
            "--metrics_config_output_dir",
            str(metrics_dir or Path.cwd() / "hybridsim_metrics"),
            "--metrics_config_run_id",
            self.run_id,
            "--no-metrics_config_write_metrics",
        ]

    def to_frontier(self) -> FrontierSimulationConfig:
        return load_frontier_config_from_cli_args(self.build_cli_args())
