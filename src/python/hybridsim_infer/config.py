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
    #: Wire KvClient (per replica) + shared KvStoreActor master.
    enable_kv_client: bool = False
    #: Floor on KV transfer TimeoutKernel duration (seconds).
    kv_transfer_s: float = 1e-4
    #: Simulated interconnect bandwidth for KV pull/push duration.
    kv_bandwidth_gbps: float = 50.0
    #: Bytes per token used when estimating transfer time.
    kv_bytes_per_token: float = 16.0
    #: Remote KV store capacity in blocks.
    kv_store_blocks: int = 4096
    #: ``store`` (MooncakeStore-style) or ``p2p`` (fixed-address lookup + Decode RDMA sim).
    kv_mode: str = "store"
    #: Fire-and-forget lookup + ReplyMsg (Store mode); pending ≈ vLLM ``None``.
    kv_lookup_async: bool = False
    #: Simulated lookup RTT before Store async reply (seconds).
    kv_lookup_rtt_s: float = 1e-3
    #: Prefill replica id when ``kv_mode=p2p``.
    kv_p2p_prefill_replica: int = 0
    #: Decode replica id when ``kv_mode=p2p`` (also used as fixed lookup location).
    kv_p2p_decode_replica: int = 1
