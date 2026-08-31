"""Forward-trace graph: contextvar-backed op list → OperatorDAG."""

from __future__ import annotations

from contextvars import ContextVar, Token

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    OperatorDAG,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    Op,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
    Tensor,
)

_GRAPH: ContextVar[Graph | None] = ContextVar("hybridsim_mock_graph", default=None)


class Graph:
    """Collects mock Ops while a ``Transformer.forward`` is running."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self._token: Token[Graph | None] | None = None

    def add(self, op: Op, inputs: list[Tensor], out_shape: tuple[int, ...]) -> Tensor:
        deps: list[int] = []
        seen: set[int] = set()
        for t in inputs:
            pid = t.producer
            if pid is not None and pid not in seen:
                seen.add(pid)
                deps.append(pid)
        op.deps = deps
        idx = len(self.ops)
        self.ops.append(op)
        dtype = int(getattr(op, "dtype_bytes", 0) or 0)
        if dtype <= 0 and inputs:
            dtype = int(inputs[0].dtype_bytes)
        if dtype <= 0:
            dtype = 2
        return Tensor(
            shape=tuple(int(d) for d in out_shape),
            producer=idx,
            dtype_bytes=dtype,
        )

    def to_operator_dag(self) -> OperatorDAG:
        dag = OperatorDAG()
        for op in self.ops:
            dag.add(op)
        return dag

    def __enter__(self) -> Graph:
        self._token = _GRAPH.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _GRAPH.reset(self._token)
            self._token = None


def current_graph() -> Graph:
    g = _GRAPH.get()
    if g is None:
        raise RuntimeError("mock Graph is not active; wrap forward in `with Graph()`")
    return g
