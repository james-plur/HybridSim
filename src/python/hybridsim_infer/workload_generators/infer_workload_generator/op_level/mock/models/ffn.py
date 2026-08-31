"""Mock FFN: TP-split up/down weights then GEMM + act Mem."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
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
from hybridsim_infer.workload_generators.types import FfnActivation


def _gated(activation: FfnActivation) -> bool:
    return activation in (FfnActivation.SILU, FfnActivation.SWIGLU)


class FFN(Module):
    def __init__(self, model: ModelConfig, parallel: ParallelConfig) -> None:
        super().__init__()
        h = max(1, int(model.hidden_size))
        i = max(1, int(model.intermediate_size))
        up_mats = 2 if _gated(model.resolved_ffn_activation()) else 1
        self.up = Shape([h, i * up_mats])
        self.down = Shape([i, h])
        self.model = model
        moe_tp = parallel.resolved_moe_tp()
        self.tp = moe_tp if moe_tp > 1 else parallel.resolved_attn_tp()

    def forward(self, x: Tensor, *, layer_id: int, batch: BatchFeatures) -> Tensor:
        _ = batch
        lid = int(layer_id)
        up = self.up.clone().split(1, self.tp)
        down = self.down.clone().split(0, self.tp)
        y = GemmOp.apply(x, up, name=f"L{lid}.gemm_up")
        y = MemOp.apply(y, [y.shape, y.shape], name=f"L{lid}.mlp_act", out_shape=y.shape)
        i_local = max(1, int(down.dims[0]))
        s = int(x.shape[0])
        h = int(down.dims[1])
        return GemmOp.apply(
            y,
            (s, i_local),
            down,
            name=f"L{lid}.gemm_down",
            dtype_bytes=x.dtype_bytes,
            out_shape=(s, h),
        )
