"""Configuration for NO_NETWORK inference simulation."""

from __future__ import annotations

from dataclasses import dataclass

from hybridsim.config import SimulationConfig


@dataclass
class InferenceConfig(SimulationConfig):
    """NO_NETWORK monolithic / multi-replica inference config."""

    #: Topology: ``monolith`` (all replicas equal) or ``pd`` (Prefill+Decode pools).
    cluster_type: str = "monolith"
    #: Monolith: total replicas. Ignored when ``cluster_type=pd`` (derived from P+D).
    num_replicas: int = 1
    #: PD: number of Prefill-pool replicas (ids ``0 .. Np-1``).
    num_prefill_replicas: int = 1
    #: PD: number of Decode-pool replicas (ids ``Np .. Np+Nd-1``).
    num_decode_replicas: int = 1
    #: Delay between StepMsg ticks (avoids zero-time busy loop when idle work remains).
    step_interval: float = 1e-3
    #: Dummy TimeoutKernel duration when ``duration_mode="fixed"``.
    dummy_exec_s: float = 0.05
    #: ``fixed``, ``token_proportional``, ``predict`` (Frontier RF), or ``analytical``.
    duration_mode: str = "fixed"
    prefill_s_per_token: float = 1e-4
    decode_s_per_token: float = 1e-3
    duration_base_s: float = 0.0
    #: When ``duration_mode=predict``: dense MoE flag for Frontier ``Batch.is_moe``.
    frontier_is_moe: bool = False
    #: When ``duration_mode=predict``: replica id stamped on adapted Frontier ``Batch``.
    frontier_replica_id: int = 0
    #: Injected Frontier predictor for ``predict`` mode (not serialized; set by caller).
    frontier_predictor: object | None = None
    #: Injected Frontier ``ClusterType`` for ``predict`` mode (default MONOLITHIC).
    frontier_cluster_type: object | None = None
    #: ``AnalyticalConfig`` for analytical DAG / device / network (or None → defaults).
    analytical_config: object | None = None
    #: Model preset id (e.g. ``llama-3.1-8b`` / ``deepseek-v3``). Shared across all
    #: ``duration_mode`` values: injects ``ModelConfig`` into analytical DAG and KV
    #: transfer. For ``predict``, RF timing still comes from ``frontier_predictor``;
    #: preset and RF should describe the same model family.
    model_preset: str | None = None
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
    #: Wire KvClient (per replica) + shared KvStoreActor (works for monolith and PD).
    enable_kv_client: bool = False
    #: Floor on KV transfer TimeoutKernel duration (seconds).
    kv_transfer_s: float = 1e-4
    #: Simulated interconnect bandwidth for KV pull/push duration.
    kv_bandwidth_gbps: float = 50.0
    #: Bytes per token used when estimating transfer time (fallback without model).
    kv_bytes_per_token: float = 16.0
    #: Fixed latency α (seconds) for KV transfer α-β model.
    kv_latency_s: float = 0.0
    #: Remote KV store DRAM capacity in blocks (``<=0`` → unlimited).
    kv_store_blocks: int = 4096
    #: SSD tier capacity in blocks (``<=0`` → SSD disabled).
    kv_store_ssd_blocks: int = 0
    #: Effective NVMe sequential-read bandwidth for SSD→DRAM staging (GB/s).
    kv_ssd_bandwidth_gbps: float = 6.0
    #: Fixed latency α (seconds) for SSD→DRAM staging.
    kv_ssd_latency_s: float = 0.0
    #: Fire-and-forget Store lookup + ReplyMsg; pending ≈ vLLM ``None``.
    kv_lookup_async: bool = False
    #: Simulated lookup / Prefill control-plane RTT (seconds).
    kv_lookup_rtt_s: float = 1e-3
    #: Max concurrent Worker batches per replica (async pipeline depth).
    #: Occupancy held from submit until BatchEnd is fully handled.
    max_inflight_batches: int = 1

    def resolved_cluster_type(self) -> str:
        ct = (self.cluster_type or "monolith").lower().strip()
        if ct in ("monolith", "pd"):
            return ct
        return "monolith"

    def resolved_num_replicas(self) -> int:
        if self.resolved_cluster_type() == "pd":
            return int(self.num_prefill_replicas) + int(self.num_decode_replicas)
        return int(self.num_replicas)

    def pd_pools(self) -> tuple[list[int], list[int]]:
        np_ = max(1, int(self.num_prefill_replicas))
        nd_ = max(1, int(self.num_decode_replicas))
        prefill = list(range(np_))
        decode = list(range(np_, np_ + nd_))
        return prefill, decode
