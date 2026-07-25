"""Configuration for NO_NETWORK inference simulation."""

from __future__ import annotations

from dataclasses import dataclass

from hybridsim.config import SimulationConfig


@dataclass
class InferenceConfig(SimulationConfig):
    """NO_NETWORK monolithic / multi-replica inference config."""

    num_replicas: int = 1
    #: Delay between StepMsg ticks (avoids zero-time busy loop when idle work remains).
    step_interval: float = 1e-3
    #: Dummy TimeoutKernel duration when ``duration_mode="fixed"``.
    dummy_exec_s: float = 0.05
    #: ``fixed`` or ``token_proportional`` (fake GPU time ∝ scheduled tokens).
    duration_mode: str = "fixed"
    prefill_s_per_token: float = 1e-4
    decode_s_per_token: float = 1e-3
    duration_base_s: float = 0.0
    #: Cap on tokens scheduled for one request's prefill chunk in a step.
    tokens_per_step: int = 8
    #: Decode tokens scheduled per request per step (vLLM-like default: 1).
    decode_tokens_per_step: int = 1
    #: vLLM-style per-step token budget across all requests (0 → unlimited).
    max_num_scheduled_tokens: int = 64
    #: Max concurrent running requests.
    max_num_running_reqs: int = 32
    #: Long-prefill threshold; 0 means use ``tokens_per_step``.
    long_prefill_token_threshold: int = 0
    #: Match vLLM ``scheduler_reserve_full_isl``: admit only if full sequence fits.
    reserve_full_isl: bool = True
    #: Local token-list prefix cache (not vLLM APC hashes). Off by default.
    enable_prefix_caching: bool = False
    #: Schedule backend: ``vllm`` (more via ``FrameworkFactory.register``).
    framework: str = "vllm"
    num_gpu_blocks: int = 1024
    block_size: int = 16
    #: Wire KvStore + KvClientEngine into the topology.
    enable_kv_client: bool = False
    #: Dummy KV transfer TimeoutKernel duration (seconds).
    kv_transfer_s: float = 0.01
    #: Remote KV store capacity in blocks.
    kv_store_blocks: int = 4096
