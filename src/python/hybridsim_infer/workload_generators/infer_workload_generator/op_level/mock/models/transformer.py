"""Top-level mock Transformer: PP boundaries + per-stage decoder layers."""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm_names import (
    COMM_PP_SEND_RECV,
)
from hybridsim_infer.workload_generators.types import ensure_attn_variant_supported
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    OperatorDAG,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm import (
    CommContext,
    comm_ctx,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.graph import (
    Graph,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.decoder import (
    DecoderLayer,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.module import (
    Module,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
    Tensor,
)


class Transformer(Module):
    def __init__(self, model: ModelConfig, parallel: ParallelConfig) -> None:
        super().__init__()
        ensure_attn_variant_supported(model.resolved_attn_variant())
        self.model = model
        self.parallel = parallel
        self.pp = max(1, int(parallel.pp_size))
        self.stage = max(0, min(int(parallel.pp_stage), self.pp - 1))
        n_layers = parallel.layers_on_stage(model.num_layers)
        stride = max(1, (int(model.num_layers) + self.pp - 1) // self.pp)
        self.layers: list[DecoderLayer] = []
        for local_i in range(n_layers):
            layer_id = self.stage * stride + local_i
            use_moe = model.layer_is_moe(layer_id)
            layer = DecoderLayer(
                model, parallel, layer_id=layer_id, use_moe=use_moe
            )
            setattr(self, f"layer_{local_i}", layer)
            self.layers.append(layer)

    def forward(self, x: Tensor, batch: BatchFeatures) -> Tensor:
        if self.pp > 1 and self.stage > 0:
            x = comm_ctx.p2p(
                x,
                name=f"L{self.stage * 1000}.{COMM_PP_SEND_RECV}",
            )
        for layer in self.layers:
            x = layer(x, batch)
        if self.pp > 1 and self.stage < self.pp - 1:
            x = comm_ctx.p2p(
                x,
                name=f"L{self.stage * 1000 + len(self.layers)}.{COMM_PP_SEND_RECV}",
            )
        return x


def build_operator_dag(
    *,
    model: ModelConfig,
    parallel: ParallelConfig,
    batch: BatchFeatures,
) -> OperatorDAG:
    """Run mock ``Transformer.forward`` and return the recorded OperatorDAG."""
    tokens = max(0, int(batch.num_tokens))
    hidden = max(1, int(model.hidden_size))
    x = Tensor(
        shape=(tokens, hidden),
        producer=None,
        dtype_bytes=max(1, int(model.dtype_bytes)),
    )
    graph = Graph()
    comm = CommContext()
    with graph, comm:
        Transformer(model, parallel)(x, batch)
    return graph.to_operator_dag()
