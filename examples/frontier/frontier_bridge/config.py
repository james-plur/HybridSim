"""Frontier-specific simulation configs (extend platform SimulationConfig)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from frontier.config import SimulationConfig as FrontierSimulationConfig

from hybridsim.config import SimulationConfig

if TYPE_CHECKING:
    from frontier_bridge.context import ReplicaSchedulerKind


def frontier_root() -> Path:
    return Path(os.environ.get("FRONTIER_ROOT", "/home/y_luchenda/Frontier"))


def load_frontier_config_from_cli_args(cli_args: list[str]) -> FrontierSimulationConfig:
    """Parse Frontier ``SimulationConfig`` from CLI-style argument list."""
    import sys

    old_argv = sys.argv
    previous_cwd = os.getcwd()
    root = frontier_root()
    try:
        if root.exists():
            os.chdir(root)
        sys.argv = ["hybridsim"] + list(cli_args)
        return FrontierSimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
        os.chdir(previous_cwd)


@dataclass
class ArchitectureConfig(SimulationConfig):
    """Config for Frontier architecture cases (co-location / PDD, online/offline)."""

    frontier: FrontierSimulationConfig | None = None

    def to_frontier(self) -> FrontierSimulationConfig:
        if self.frontier is None:
            raise ValueError("ArchitectureConfig.frontier is required")
        return self.frontier

    @classmethod
    def from_cli_args(cls, argv: Sequence[str] | None = None) -> ArchitectureConfig:
        import sys

        raw = list(sys.argv[1:] if argv is None else argv)
        common, remaining = cls.parse_common_cli_args(raw)
        frontier = load_frontier_config_from_cli_args(remaining)
        return cls(frontier=frontier, **common)


@dataclass
class MonolithicConfig(SimulationConfig):
    """Config for a single MONOLITHIC cluster smoke / demo run."""

    replica_scheduler_kind: Optional["ReplicaSchedulerKind"] = None
    num_replicas: int = 1
    attn_data_parallel_size: int = 1
    dummy_execution_time_ms: float = 100.0
    metrics_output_dir: Optional[Path] = None
    run_id: str = "monolithic_smoke"

    def __post_init__(self) -> None:
        if self.trace_output_dir is None and self.metrics_output_dir is not None:
            self.trace_output_dir = self.metrics_output_dir

    def build_cli_args(self) -> list[str]:
        from frontier_bridge.context import ReplicaSchedulerKind

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

    @classmethod
    def from_cli_args(cls, argv: Sequence[str] | None = None) -> MonolithicConfig:
        # Monolithic knobs are dataclass fields; CLI for Frontier is built in to_frontier().
        common, _ = cls.parse_common_cli_args(argv or [])
        return cls(**common)
