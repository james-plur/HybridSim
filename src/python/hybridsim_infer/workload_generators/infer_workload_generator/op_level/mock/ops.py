"""Shape-only primitive ops recorded during mock ``Module.forward``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence, Union

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    CommCollective,
    OperatorKind,
    collective_volume_factor,
)

if TYPE_CHECKING:
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.shape import (
        Shape,
    )
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
        Tensor,
    )

_ShapeLike = Union[tuple[int, ...], list[int], "Shape"]


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= max(0, int(dim))
    return n


def _as_tuple(shape: _ShapeLike) -> tuple[int, ...]:
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.shape import (
        Shape,
    )

    if isinstance(shape, Shape):
        return shape.as_tuple()
    return tuple(int(d) for d in shape)


def _as_deps(x_or_deps: Tensor | list[Tensor]) -> list[Tensor]:
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
        Tensor,
    )

    if isinstance(x_or_deps, Tensor):
        return [x_or_deps]
    return list(x_or_deps)


@dataclass
class Op:
    """Leaf node: operand shapes only; duration comes from AnalyticAnalyzer."""

    name: str
    deps: list[int] = field(default_factory=list)

    @property
    def kind(self) -> OperatorKind:
        raise NotImplementedError

    def features(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class MemOp(Op):
    """Memory-bound primitive: bytes = dtype * sum(numel(shapes))."""

    shapes: tuple[tuple[int, ...], ...] = ()
    dtype_bytes: int = 2

    @property
    def kind(self) -> OperatorKind:
        return OperatorKind.MEM

    def features(self) -> dict[str, Any]:
        elems = sum(numel(s) for s in self.shapes)
        return {"flops": 0.0, "bytes": float(self.dtype_bytes * elems)}

    @classmethod
    def apply(
        cls,
        x_or_deps: Tensor | list[Tensor],
        shapes: Sequence[_ShapeLike],
        *,
        name: str,
        out_shape: tuple[int, ...] | None = None,
        dtype_bytes: int | None = None,
    ) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.graph import (
            current_graph,
        )

        deps = _as_deps(x_or_deps)
        if not deps:
            raise ValueError("MemOp.apply requires at least one dependency tensor")
        db = int(dtype_bytes) if dtype_bytes is not None else int(deps[0].dtype_bytes)
        recorded = tuple(_as_tuple(s) for s in shapes)
        out = out_shape if out_shape is not None else deps[0].shape
        op = cls(name=name, shapes=recorded, dtype_bytes=db)
        return current_graph().add(op, deps, out)


@dataclass
class GemmOp(Op):
    """Standard GEMM: A[M,K] @ B[K,N]."""

    a_shape: tuple[int, ...] = ()
    b_shape: tuple[int, ...] = ()
    dtype_bytes: int = 2

    @property
    def kind(self) -> OperatorKind:
        return OperatorKind.GEMM

    def features(self) -> dict[str, Any]:
        a0, a1 = (
            (int(self.a_shape[0]), int(self.a_shape[1]))
            if len(self.a_shape) >= 2
            else (0, 0)
        )
        b0, b1 = (
            (int(self.b_shape[0]), int(self.b_shape[1]))
            if len(self.b_shape) >= 2
            else (0, 0)
        )
        m, k, n = a0, a1, b1
        _ = b0
        dtype = max(1, int(self.dtype_bytes))
        return {
            "flops": 2.0 * m * k * n,
            "bytes": float(dtype * (m * k + k * n + m * n)),
        }

    @classmethod
    def apply(
        cls,
        x_or_deps: Tensor | list[Tensor],
        weight_or_a: _ShapeLike,
        b: _ShapeLike | None = None,
        *,
        name: str,
        dtype_bytes: int | None = None,
        out_shape: tuple[int, ...] | None = None,
    ) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.graph import (
            current_graph,
        )
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
            Tensor,
        )

        if isinstance(x_or_deps, Tensor) and b is None:
            a = _as_tuple(x_or_deps.shape)
            b_t = _as_tuple(weight_or_a)
            if len(a) != 2 or len(b_t) != 2:
                raise ValueError(f"GEMM expects rank-2 shapes, got {a} @ {b_t}")
            if a[1] != b_t[0]:
                raise ValueError(f"GEMM inner-dim mismatch {a} @ {b_t}")
            op = cls(
                name=name,
                a_shape=a,
                b_shape=b_t,
                dtype_bytes=x_or_deps.dtype_bytes,
            )
            return current_graph().add(op, [x_or_deps], (a[0], b_t[1]))

        if b is None:
            raise TypeError("GemmOp.apply with explicit A requires b (the B matrix)")
        deps = _as_deps(x_or_deps)
        a = _as_tuple(weight_or_a)
        b_t = _as_tuple(b)
        if len(a) != 2 or len(b_t) != 2:
            raise ValueError(f"GEMM expects rank-2 shapes, got {a} @ {b_t}")
        out = out_shape if out_shape is not None else (a[0], b_t[1])
        db = (
            int(dtype_bytes)
            if dtype_bytes is not None
            else (int(deps[0].dtype_bytes) if deps else 2)
        )
        op = cls(name=name, a_shape=a, b_shape=b_t, dtype_bytes=db)
        return current_graph().add(op, deps, out)


@dataclass
class CommOp(Op):
    """Collective / P2P. ``num_ranks`` is an attribute of this collective instance."""

    payload_shape: tuple[int, ...] = ()
    dtype_bytes: int = 2
    collective: CommCollective = CommCollective.ALLREDUCE
    num_ranks: int = 1

    @property
    def kind(self) -> OperatorKind:
        return OperatorKind.COMM

    def features(self) -> dict[str, Any]:
        ranks = max(1, int(self.num_ranks))
        payload = numel(self.payload_shape) * max(1, int(self.dtype_bytes))
        factor = collective_volume_factor(self.collective, ranks)
        return {
            "payload_bytes": payload,
            "num_ranks": ranks,
            "volume_factor": factor,
            "collective": self.collective.value,
            "flops": 0.0,
            "bytes": 0.0,
        }

    @classmethod
    def apply(
        cls,
        x: Tensor,
        *,
        name: str,
        collective: CommCollective,
        num_ranks: int,
    ) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.graph import (
            current_graph,
        )

        ranks = max(1, int(num_ranks))
        if ranks <= 1:
            return x
        op = cls(
            name=name,
            payload_shape=tuple(int(d) for d in x.shape),
            dtype_bytes=int(x.dtype_bytes),
            collective=collective,
            num_ranks=ranks,
        )
        return current_graph().add(op, [x], x.shape)
