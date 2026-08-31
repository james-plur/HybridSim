"""Mock Module / Shape / forward DAG construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HYBRIDSIM_ROOT = Path(__file__).resolve().parents[1]
_PY = _HYBRIDSIM_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.workload_generators.configs import ModelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    BatchFeatures,
    BatchPhase,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.comm_names import (
    COMM_ATTN_TP_ALLREDUCE,
    COMM_MLP_TP_ALLREDUCE,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.expect import (
    expected_layer_primitives,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.models.transformer import (
    Transformer,
    build_operator_dag,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.module import (
    Module,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.shape import (
    Shape,
)
from hybridsim_infer.workload_generators.types import AttnVariant


def _prefill_feats(tokens: int = 16) -> BatchFeatures:
    return BatchFeatures(
        phase=BatchPhase.PREFILL,
        num_tokens=tokens,
        num_prefill_tokens=tokens,
        num_decode_tokens=0,
        batch_size=1,
        prefill_chunk_lens=[tokens],
        prefill_cached_lens=[0],
    )


class TestShape(unittest.TestCase):
    def test_split_does_not_mutate_original(self) -> None:
        w = Shape([4096, 4096])
        local = w.clone().split(1, 4)
        self.assertEqual(local.dims, [4096, 1024])
        self.assertEqual(w.dims, [4096, 4096])


class TestModuleNesting(unittest.TestCase):
    def test_named_children(self) -> None:
        class Child(Module):
            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = Child()
                self.right = Child()

        p = Parent()
        names = [n for n, _ in p.named_children()]
        self.assertEqual(names, ["left", "right"])


class TestMockForwardDag(unittest.TestCase):
    def test_dense_layer_matches_primitives(self) -> None:
        model = ModelConfig(num_layers=1, attn_variant=AttnVariant.GQA)
        dag = build_operator_dag(
            model=model,
            parallel=ParallelConfig(),
            batch=_prefill_feats(),
        )
        expect = expected_layer_primitives(
            attn_variant=AttnVariant.GQA,
            phase=BatchPhase.PREFILL,
            is_moe=False,
            num_prefill=1,
        )
        self.assertEqual(dag.op_names(), expect)

    def test_tp_inserts_allreduce(self) -> None:
        dag = build_operator_dag(
            model=ModelConfig(num_layers=1),
            parallel=ParallelConfig(tp_size=4),
            batch=_prefill_feats(),
        )
        names = dag.op_names()
        self.assertIn(COMM_ATTN_TP_ALLREDUCE, names)
        self.assertIn(COMM_MLP_TP_ALLREDUCE, names)

    def test_transformer_registers_layers(self) -> None:
        model = Transformer(ModelConfig(num_layers=2), ParallelConfig())
        child_names = [n for n, _ in model.named_children()]
        self.assertIn("layer_0", child_names)
        self.assertIn("layer_1", child_names)

    def test_first_k_dense_replace(self) -> None:
        model = ModelConfig(
            num_layers=2,
            is_moe=True,
            first_k_dense_replace=1,
            num_experts=8,
            num_experts_per_tok=2,
        )
        dag = build_operator_dag(
            model=model,
            parallel=ParallelConfig(),
            batch=_prefill_feats(),
        )
        names = dag.op_names()
        self.assertIn("gemm_up", names)
        self.assertIn("gemm_moe_up", names)


if __name__ == "__main__":
    unittest.main()
