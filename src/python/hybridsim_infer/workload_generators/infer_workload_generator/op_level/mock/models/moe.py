"""Mock MoE + shared-expert blocks with EP dispatch/combine."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm_names import (
    COMM_EP_COMBINE,
    COMM_EP_DISPATCH,
    COMM_MOE_TP_ALLGATHER,
    COMM_MOE_TP_ALLREDUCE,
    COMM_SHARE_EXPERT_TP_ALLREDUCE,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm import (
    comm_ctx,
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


class ShareExpert(Module):
    def __init__(self, model: ModelConfig, parallel: ParallelConfig) -> None:
        super().__init__()
        h = max(1, int(model.hidden_size))
        i = max(1, int(model.share_expert_dim))
        self.up = Shape([h, i * 2])
        self.down = Shape([i, h])
        self.model = model
        self.tp = parallel.resolved_moe_tp()

    def forward(self, x: Tensor, *, layer_id: int, batch: BatchFeatures) -> Tensor:
        _ = batch
        lid = int(layer_id)
        up = self.up.clone().split(1, self.tp)
        down = self.down.clone().split(0, self.tp)
        y = GemmOp.apply(x, up, name=f"L{lid}.gemm_share_up")
        y = MemOp.apply(
            y, [y.shape, y.shape], name=f"L{lid}.share_act", out_shape=y.shape
        )
        i_local = max(1, int(down.dims[0]))
        s = int(x.shape[0])
        h = int(down.dims[1])
        return GemmOp.apply(
            y,
            (s, i_local),
            down,
            name=f"L{lid}.gemm_share_down",
            dtype_bytes=x.dtype_bytes,
            out_shape=(s, h),
        )


class MoE(Module):
    def __init__(self, model: ModelConfig, parallel: ParallelConfig) -> None:
        super().__init__()
        self.model = model
        self.tp = parallel.resolved_moe_tp()
        self.ep = max(1, int(parallel.ep_size))
        h = max(1, int(model.hidden_size))
        e = max(1, int(model.num_experts))
        i = max(1, int(model.intermediate_size))
        self.gate = Shape([h, e])
        self.up = Shape([h, i * 2])
        self.down = Shape([i, h])
        self.share = ShareExpert(model, parallel) if model.has_share_expert() else None

    def forward(self, x: Tensor, *, layer_id: int, batch: BatchFeatures) -> Tensor:
        lid = int(layer_id)
        s = int(x.shape[0])
        h = int(x.shape[1])
        e = max(1, int(self.model.num_experts))
        k = max(1, int(self.model.num_experts_per_tok))
        x = GemmOp.apply(x, self.gate, name=f"L{lid}.moe_gating")
        x = MemOp.apply(
            x,
            [(s, e), (s, e)],
            name=f"L{lid}.moe_topk",
            out_shape=(s, h),
        )
        x = comm_ctx.allgather(
            x,
            name=f"L{lid}.{COMM_MOE_TP_ALLGATHER}",
            num_ranks=self.tp,
        )
        if self.ep > 1:
            x = comm_ctx.dispatch(
                x,
                name=f"L{lid}.{COMM_EP_DISPATCH}",
                num_ranks=self.ep,
            )
        x = MemOp.apply(
            x,
            [(s, h), (s, h)],
            name=f"L{lid}.moe_shuffle",
            out_shape=x.shape,
        )
        active = s * k
        up = self.up.clone().split(1, self.tp)
        down = self.down.clone().split(0, self.tp)
        y = GemmOp.apply(
            x,
            (active, h),
            up,
            name=f"L{lid}.gemm_moe_up",
            dtype_bytes=x.dtype_bytes,
        )
        i_local = max(1, int(down.dims[0]))
        y = GemmOp.apply(
            y,
            (active, i_local),
            down,
            name=f"L{lid}.gemm_moe_down",
            dtype_bytes=x.dtype_bytes,
            out_shape=(s, h),
        )
        if self.ep > 1:
            y = comm_ctx.combine(
                y,
                name=f"L{lid}.{COMM_EP_COMBINE}",
                num_ranks=self.ep,
            )
        if self.share is not None:
            y = self.share(y, layer_id=lid, batch=batch)
            y = comm_ctx.allreduce(
                y,
                name=f"L{lid}.{COMM_SHARE_EXPERT_TP_ALLREDUCE}",
                num_ranks=self.tp,
            )
        y = comm_ctx.allreduce(
            y,
            name=f"L{lid}.{COMM_MOE_TP_ALLREDUCE}",
            num_ranks=self.tp,
        )
        return y
