"""Extensible fused kernels: subclass ``FusedOp`` and implement cost + out shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    OperatorKind,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    Op,
)

if TYPE_CHECKING:
    from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
        Tensor,
    )


@dataclass
class FusedOp(Op):
    """Fused compute kernel. Analyzer times these with Roofline (``kind=FUSED``)."""

    dtype_bytes: int = 2

    @property
    def kind(self) -> OperatorKind:
        return OperatorKind.FUSED

    def infer_out_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def record(self, deps: Tensor | list[Tensor]) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.graph import (
            current_graph,
        )
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
            _as_deps,
        )

        return current_graph().add(self, _as_deps(deps), self.infer_out_shape())


@dataclass
class FusedAttnOp(FusedOp):
    """Dense fused attention. ``kernel`` is prefill|decode (HBM formula)."""

    q_shape: tuple[int, ...] = ()
    k_shape: tuple[int, ...] = ()
    v_shape: tuple[int, ...] = ()
    kernel: str = "prefill"

    def infer_out_shape(self) -> tuple[int, ...]:
        q, n_q, d = (int(x) for x in self.q_shape[:3])
        return (q, n_q * d)

    def features(self) -> dict[str, Any]:
        q, n_q, d = (int(x) for x in self.q_shape[:3])
        kv, n_kv, _d_k = (int(x) for x in self.k_shape[:3])
        dtype = max(1, int(self.dtype_bytes))
        flops = 4.0 * n_q * d * q * kv
        if self.kernel == "decode":
            nbytes = dtype * (q * n_q * d + q * kv * n_kv * d * 2 + q * n_q * d)
        else:
            nbytes = dtype * (q * n_q * d + kv * n_kv * d * 2 + q * n_q * d)
        return {"flops": flops, "bytes": float(nbytes)}

    @classmethod
    def apply(
        cls,
        deps: Tensor | list[Tensor],
        *,
        name: str,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        kernel: str,
        dtype_bytes: int | None = None,
    ) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
            _as_deps,
        )

        dep_list = _as_deps(deps)
        db = (
            int(dtype_bytes)
            if dtype_bytes is not None
            else (int(dep_list[0].dtype_bytes) if dep_list else 2)
        )
        op = cls(
            name=name,
            q_shape=tuple(int(x) for x in q_shape),
            k_shape=tuple(int(x) for x in k_shape),
            v_shape=tuple(int(x) for x in v_shape),
            dtype_bytes=db,
            kernel=kernel,
        )
        return op.record(dep_list)


@dataclass
class FusedMlaAttnOp(FusedOp):
    """MLA fused attention. ``q_shape`` is [q, n_q, latent]; ``kv_shape`` is [kv, latent]."""

    q_shape: tuple[int, ...] = ()
    kv_shape: tuple[int, ...] = ()
    kernel: str = "prefill"
    out_width: int = 0

    def infer_out_shape(self) -> tuple[int, ...]:
        q = int(self.q_shape[0])
        return (q, int(self.out_width))

    def features(self) -> dict[str, Any]:
        q, n_q, latent = (int(x) for x in self.q_shape[:3])
        kv = int(self.kv_shape[0])
        dtype = max(1, int(self.dtype_bytes))
        flops = 4.0 * n_q * latent * q * kv
        if self.kernel == "decode":
            nbytes = dtype * (q * n_q * latent + q * kv * latent)
        else:
            nbytes = dtype * (q * n_q * latent + kv * latent + q * n_q * latent)
        return {"flops": flops, "bytes": float(nbytes)}

    @classmethod
    def apply(
        cls,
        deps: Tensor | list[Tensor],
        *,
        name: str,
        q_shape: tuple[int, ...],
        kv_shape: tuple[int, ...],
        kernel: str,
        out_width: int,
        dtype_bytes: int | None = None,
    ) -> Tensor:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
            _as_deps,
        )

        dep_list = _as_deps(deps)
        db = (
            int(dtype_bytes)
            if dtype_bytes is not None
            else (int(dep_list[0].dtype_bytes) if dep_list else 2)
        )
        op = cls(
            name=name,
            q_shape=tuple(int(x) for x in q_shape),
            kv_shape=tuple(int(x) for x in kv_shape),
            dtype_bytes=db,
            kernel=kernel,
            out_width=int(out_width),
        )
        return op.record(dep_list)
