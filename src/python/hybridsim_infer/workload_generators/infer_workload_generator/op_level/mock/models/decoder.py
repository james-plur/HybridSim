"""One transformer decoder layer (norm / attn / FFN or MoE / residual)."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm_names import (
    COMM_ATTN_TP_ALLREDUCE,
    COMM_MLP_TP_ALLREDUCE,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm import (
    comm_ctx,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.attention import (
    Attention,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.ffn import (
    FFN,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.moe import (
    MoE,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.module import (
    Module,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    MemOp,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
    Tensor,
)


class DecoderLayer(Module):
    def __init__(
        self,
        model: ModelConfig,
        parallel: ParallelConfig,
        *,
        layer_id: int,
        use_moe: bool,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.model = model
        self.parallel = parallel
        self.fused = bool(model.fused_add_norm)
        self.attn = Attention(model, parallel)
        self.ffn: FFN | None = None
        self.moe: MoE | None = None
        if use_moe:
            self.moe = MoE(model, parallel)
        else:
            self.ffn = FFN(model, parallel)

    def _norm_or_residual(self, x: Tensor, name: str) -> Tensor:
        # Three copies of [s, h] → bytes = 3 * s * h * dtype (norm / residual).
        return MemOp.apply(
            x,
            [x.shape, x.shape, x.shape],
            name=f"L{self.layer_id}.{name}",
            out_shape=x.shape,
        )

    def forward(self, x: Tensor, batch: BatchFeatures) -> Tensor:
        lid = self.layer_id
        x = self._norm_or_residual(x, "input_layernorm")
        x = self.attn(x, layer_id=lid, batch=batch)
        x = comm_ctx.allreduce(
            x,
            name=f"L{lid}.{COMM_ATTN_TP_ALLREDUCE}",
            num_ranks=self.parallel.resolved_attn_tp(),
        )
        if not self.fused:
            x = self._norm_or_residual(x, "add_attn_residual")
        x = self._norm_or_residual(x, "post_attention_layernorm")
        if self.moe is not None:
            x = self.moe(x, layer_id=lid, batch=batch)
        else:
            assert self.ffn is not None
            x = self.ffn(x, layer_id=lid, batch=batch)
            mlp_tp = max(
                self.parallel.resolved_moe_tp(), self.parallel.resolved_attn_tp()
            )
            x = comm_ctx.allreduce(
                x,
                name=f"L{lid}.{COMM_MLP_TP_ALLREDUCE}",
                num_ranks=mlp_tp,
            )
        if not self.fused:
            x = self._norm_or_residual(x, "add_ffn_residual")
        return x
