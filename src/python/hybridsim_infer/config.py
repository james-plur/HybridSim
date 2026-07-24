"""Configuration for NO_NETWORK inference simulation."""

from __future__ import annotations

from dataclasses import dataclass

from hybridsim.config import SimulationConfig


@dataclass
class InferenceConfig(SimulationConfig):
    """NO_NETWORK monolithic / multi-replica inference skeleton config."""

    num_replicas: int = 1
    #: Delay between StepMsg ticks (avoids zero-time busy loop when idle work remains).
    step_interval: float = 1e-3
    #: Dummy TimeoutKernel duration per scheduled batch (seconds).
    dummy_exec_s: float = 0.05
    #: Tokens advanced per dummy batch step (prefill+decode progress).
    tokens_per_step: int = 8
    num_gpu_blocks: int = 1024
    block_size: int = 16
