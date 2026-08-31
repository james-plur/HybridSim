"""Mock Attention: TP-split weights, GEMM + fused attn from operand shapes."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.fused import (
    FusedAttnOp,
    FusedMlaAttnOp,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.module import (
    Module,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    GemmOp,
    MemOp,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.shape import (
    Shape,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
    Tensor,
)
from hybridsim_infer.workload_generators.types import AttnVariant, ensure_attn_variant_supported


def _dense_head_counts(
    model: ModelConfig, tp: int
) -> tuple[int, int, int]:
    n_q = max(1, int(model.num_q_heads))
    n_kv = max(1, int(model.num_kv_heads))
    variant = model.resolved_attn_variant()
    if variant is AttnVariant.MHA:
        n_kv = n_q
    elif variant is AttnVariant.MQA:
        n_kv = 1
    d = max(1, int(model.get_head_dim()))
    n_q_l = max(1, n_q // max(1, tp))
    n_kv_l = max(1, n_kv // max(1, tp))
    return n_q_l, n_kv_l, d


class Attention(Module):
    def __init__(self, model: ModelConfig, parallel: ParallelConfig) -> None:
        super().__init__()
        ensure_attn_variant_supported(model.resolved_attn_variant())
        self.model = model
        self.tp = parallel.resolved_attn_tp()
        h = max(1, int(model.hidden_size))
        n_q = max(1, int(model.num_q_heads))
        n_kv = max(1, int(model.num_kv_heads))
        variant = model.resolved_attn_variant()
        if variant is AttnVariant.MHA:
            n_kv = n_q
        elif variant is AttnVariant.MQA:
            n_kv = 1
        d = max(1, int(model.get_head_dim()))
        # Packed like vLLM QKVParallelLinear: one GEMM, then split Q/K/V.
        self.q = Shape([h, n_q * d])
        self.k = Shape([h, n_kv * d])
        self.v = Shape([h, n_kv * d])
        self.o = Shape([n_q * d, h])

    def forward(self, x: Tensor, *, layer_id: int, batch: BatchFeatures) -> Tensor:
        variant = self.model.resolved_attn_variant()
        if variant in (AttnVariant.MLA, AttnVariant.DSA):
            return self._forward_mla(x, layer_id=layer_id, batch=batch)
        return self._forward_dense(x, layer_id=layer_id, batch=batch)

    def _forward_dense(
        self, x: Tensor, *, layer_id: int, batch: BatchFeatures
    ) -> Tensor:
        lid = int(layer_id)
        q_w = self.q.clone().split(1, self.tp)
        k_w = self.k.clone().split(1, self.tp)
        v_w = self.v.clone().split(1, self.tp)
        o_w = self.o.clone().split(0, self.tp)
        n_q_l, n_kv_l, d = _dense_head_counts(self.model, self.tp)
        h = int(x.shape[1])
        qkv_w = Shape([h, q_w.dims[1] + k_w.dims[1] + v_w.dims[1]])
        qkv = GemmOp.apply(x, qkv_w, name=f"L{lid}.gemm_qkv")
        s = int(x.shape[0])
        q_3 = (s, n_q_l, d)
        k_3 = (s, n_kv_l, d)
        qkv = MemOp.apply(
            qkv, [q_3, q_3, k_3, k_3], name=f"L{lid}.rope", out_shape=qkv.shape
        )
        ys: list[Tensor] = []
        for chunk, cached in batch.iter_prefill_attn_pairs():
            ctx = max(int(chunk), int(cached) + int(chunk))
            ys.append(
                FusedAttnOp.apply(
                    qkv,
                    name=f"L{lid}.fused_attn",
                    q_shape=(int(chunk), n_q_l, d),
                    k_shape=(ctx, n_kv_l, d),
                    v_shape=(ctx, n_kv_l, d),
                    dtype_bytes=x.dtype_bytes,
                    kernel="prefill",
                )
            )
        for tokens_i, ctx_i in batch.iter_decode_attn_pairs():
            ys.append(
                FusedAttnOp.apply(
                    qkv,
                    name=f"L{lid}.fused_attn",
                    q_shape=(int(tokens_i), n_q_l, d),
                    k_shape=(int(ctx_i), n_kv_l, d),
                    v_shape=(int(ctx_i), n_kv_l, d),
                    dtype_bytes=x.dtype_bytes,
                    kernel="decode",
                )
            )
        kv = MemOp.apply(
            qkv, [k_3, k_3], name=f"L{lid}.kv_cache_save", out_shape=qkv.shape
        )
        deps = (ys if ys else [qkv]) + [kv]
        return GemmOp.apply(
            deps,
            (s, n_q_l * d),
            o_w,
            name=f"L{lid}.gemm_o",
            dtype_bytes=x.dtype_bytes,
            out_shape=(s, int(x.shape[1])),
        )

    def _forward_mla(
        self, x: Tensor, *, layer_id: int, batch: BatchFeatures
    ) -> Tensor:
        lid = int(layer_id)
        model = self.model
        tp = max(1, self.tp)
        s = int(x.shape[0])
        h = max(1, int(model.hidden_size))
        n_q_l = max(1, int(model.num_q_heads) // tp)
        kv_rank = max(1, int(model.kv_lora_rank))
        rope_d = max(1, int(model.qk_rope_head_dim))
        nope_d = max(1, int(model.qk_nope_head_dim))
        v_d = max(1, int(model.v_head_dim))
        latent = kv_rank + rope_d
        q_rank = int(model.q_lora_rank) or h
        dtype = int(x.dtype_bytes)

        x = MemOp.apply(
            x, [(s, latent)], name=f"L{lid}.kv_cache_save", out_shape=x.shape
        )

        cores: list[Tensor] = []
        for chunk, cached in batch.iter_prefill_attn_pairs():
            ctx = max(int(chunk), int(cached) + int(chunk))
            kv_up = GemmOp.apply(
                x,
                (int(chunk), kv_rank),
                (kv_rank, n_q_l * (nope_d + v_d)),
                name=f"L{lid}.gemm_kv_up",
                dtype_bytes=dtype,
            )
            cores.append(
                FusedMlaAttnOp.apply(
                    kv_up,
                    name=f"L{lid}.fused_mla_attn",
                    q_shape=(int(chunk), n_q_l, latent),
                    kv_shape=(ctx, latent),
                    dtype_bytes=dtype,
                    kernel="prefill",
                    out_width=n_q_l * v_d,
                )
            )
        for tokens_i, ctx_i in batch.iter_decode_attn_pairs():
            q_a = GemmOp.apply(
                x,
                (int(tokens_i), h),
                (h, q_rank),
                name=f"L{lid}.gemm_q_lora",
                dtype_bytes=dtype,
            )
            q_b = GemmOp.apply(
                q_a,
                (int(tokens_i), q_rank),
                (q_rank, n_q_l * (nope_d + rope_d)),
                name=f"L{lid}.gemm_q_expand",
                dtype_bytes=dtype,
            )
            cores.append(
                FusedMlaAttnOp.apply(
                    q_b,
                    name=f"L{lid}.fused_mla_attn",
                    q_shape=(int(tokens_i), n_q_l, latent),
                    kv_shape=(int(ctx_i), latent),
                    dtype_bytes=dtype,
                    kernel="decode",
                    out_width=n_q_l * v_d,
                )
            )
        deps = cores if cores else [x]
        y = GemmOp.apply(
            deps,
            (s, n_q_l * v_d),
            (n_q_l * v_d, h),
            name=f"L{lid}.gemm_v_up",
            dtype_bytes=dtype,
            out_shape=(s, h),
        )
        if model.resolved_attn_variant() is AttnVariant.DSA:
            idx_hd = max(0, int(getattr(model, "index_head_dim", 0) or 0))
            idx_b = int(getattr(model, "index_dtype_bytes", 0) or 0) or dtype
            y = MemOp.apply(
                y,
                [(s, idx_hd)],
                name=f"L{lid}.indexer_cache_save",
                out_shape=y.shape,
                dtype_bytes=idx_b,
            )
        return y
