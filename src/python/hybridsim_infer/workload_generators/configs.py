"""Shared model / parallelism / device / network configs for all workload paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybridsim_infer.workload_generators.types import (
    AttnVariant,
    FfnActivation,
)


@dataclass
class ModelConfig:
    """Transformer model shape used by infer DAG construction and KV volume.

    Field names are hybridsim-native; comments note Frontier equivalents.
    """

    num_layers: int = 2
    #: Frontier ``embedding_dim``.
    hidden_size: int = 4096
    #: Frontier ``mlp_hidden_dim`` (per-expert intermediate for MoE).
    intermediate_size: int = 11008
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    #: Bytes per element (BF16/FP16 → 2).
    dtype_bytes: int = 2
    attn_variant: AttnVariant | str = AttnVariant.GQA
    ffn_activation: FfnActivation | str = FfnActivation.SILU
    #: MLA latent dims (used when attn_variant=mla / dsa).
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    #: KV volume formula: ``standard_gqa`` | ``mla`` | ``dsa_mla`` | ``deepseek_v4_hybrid``.
    kv_formula: str = ""
    #: DSA / V4 indexer dims (unused for GQA/MLA).
    index_head_dim: int = 0
    index_n_heads: int = 0
    index_topk: int = 0
    #: IndexShare: every ``index_topk_freq`` layers share one indexer (0 → all layers).
    index_topk_freq: int = 0
    index_skip_topk_offset: int = 0
    #: Indexer element size; 0 → use ``dtype_bytes``.
    index_dtype_bytes: int = 0
    #: Per-layer compress ratios for ``deepseek_v4_hybrid``.
    compress_ratios: list[int] | None = None
    #: Sliding-window width for V4 hybrid KV.
    sliding_window: int = 0
    #: Dense→MoE transition: layers ``[0, first_k_dense_replace)`` stay dense.
    first_k_dense_replace: int = 0
    #: Optional MTP / next-n draft layers (KV draft counted only if ``include_draft_kv``).
    num_nextn_predict_layers: int = 0
    #: When True, ``kv_cache.cache_bytes`` includes draft / MTP layers.
    include_draft_kv: bool = False
    #: MoE (Frontier ``MoEModelConfig``).
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    #: Shared-expert intermediate dim; 0 disables share_expert ops.
    share_expert_dim: int = 0
    #: When True, omit explicit residual add ops (fused into norms).
    fused_add_norm: bool = False

    def resolved_attn_variant(self) -> AttnVariant:
        if isinstance(self.attn_variant, AttnVariant):
            return self.attn_variant
        return AttnVariant(str(self.attn_variant).lower().strip())

    def resolved_ffn_activation(self) -> FfnActivation:
        if isinstance(self.ffn_activation, FfnActivation):
            return self.ffn_activation
        return FfnActivation(str(self.ffn_activation).lower().strip())

    def get_head_dim(self) -> int:
        return int(self.head_dim)

    def has_share_expert(self) -> bool:
        return bool(self.is_moe) and int(self.share_expert_dim) > 0

    def layer_is_moe(self, layer_id: int) -> bool:
        """Whether ``layer_id`` (global) uses MoE rather than dense FFN."""
        if not self.is_moe:
            return False
        return int(layer_id) >= max(0, int(self.first_k_dense_replace))


@dataclass
class ParallelConfig:
    """Data / tensor / pipeline / expert parallelism.

    ``tp_size`` remains the default; ``attn_tp_size`` / ``moe_tp_size`` override
    when set (>0), matching Frontier's split attention vs MoE TP.
    """

    tp_size: int = 1
    #: Frontier ``attn_tensor_parallel_size`` (0 → use ``tp_size``).
    attn_tp_size: int = 0
    #: Frontier ``moe_tensor_parallel_size`` (0 → use ``tp_size``).
    moe_tp_size: int = 0
    pp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    #: Which PP stage this replica simulates (0 .. pp_size-1).
    pp_stage: int = 0

    def resolved_attn_tp(self) -> int:
        return max(1, int(self.attn_tp_size) or int(self.tp_size))

    def resolved_moe_tp(self) -> int:
        return max(1, int(self.moe_tp_size) or int(self.tp_size))

    def layers_on_stage(self, num_layers: int) -> int:
        """Layers owned by ``pp_stage`` (even split, remainder on early stages)."""
        pp = max(1, int(self.pp_size))
        stage = max(0, min(int(self.pp_stage), pp - 1))
        base, rem = divmod(int(num_layers), pp)
        return base + (1 if stage < rem else 0)


@dataclass
class DeviceConfig:
    """GPU compute / memory peaks for Roofline."""

    #: Peak FLOP/s (e.g. A100 BF16 ≈ 312e12).
    peak_flops: float = 312e12
    #: HBM bandwidth in bytes/s (e.g. A100 ≈ 2.039e12).
    hbm_bandwidth_bps: float = 2.039e12
    #: Achieved fraction of ``peak_flops`` (kernel / MFU-style efficiency).
    #: Effective compute rate = ``peak_flops * compute_util``.
    compute_util: float = 0.6
    #: Achieved fraction of ``hbm_bandwidth_bps``.
    #: Effective bandwidth = ``hbm_bandwidth_bps * hbm_util``.
    #: Empirically ~0.6 vs Frontier H800 RF on Llama-2-7B memory-bound batches.
    hbm_util: float = 0.6

    def effective_peak_flops(self) -> float:
        u = min(1.0, max(1e-6, float(self.compute_util)))
        return max(1e-30, float(self.peak_flops) * u)

    def effective_hbm_bandwidth_bps(self) -> float:
        u = min(1.0, max(1e-6, float(self.hbm_util)))
        return max(1e-30, float(self.hbm_bandwidth_bps) * u)


@dataclass
class NetworkConfig:
    """α-β interconnect model for collective / P2P communication."""

    #: Fixed latency per collective / message (seconds).
    alpha_s: float = 1e-6
    #: Inverse bandwidth (seconds per byte).
    beta_s_per_byte: float = 1.0 / (50e9 / 8.0)  # ~50 Gbps default

    @classmethod
    def from_bandwidth(
        cls,
        *,
        latency_s: float = 1e-6,
        bandwidth_gbps: float = 50.0,
    ) -> NetworkConfig:
        bps = float(bandwidth_gbps) * 1e9 / 8.0
        return cls(alpha_s=float(latency_s), beta_s_per_byte=1.0 / bps)


@dataclass
class OpLevelConfig:
    """Bundle of configs required by OpLevelWorkloadGenerator.

    ``compute_analyzer`` / ``comm_analyzer`` select lowering strategies
    independently (see ``op_level.analyzers``).
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    #: Optional global duration multiplier (set after offline calibration).
    duration_scale: float = 1.0
    #: Compute-op lowering: ``analytic`` (Roofline / mem TimeoutKernel).
    compute_analyzer: str = "analytic"
    #: Comm-op lowering: ``analytic`` (α-β TimeoutKernel) or ``ring`` (Put/Wait).
    comm_analyzer: str = "analytic"

    @classmethod
    def default(cls) -> OpLevelConfig:
        return cls()
