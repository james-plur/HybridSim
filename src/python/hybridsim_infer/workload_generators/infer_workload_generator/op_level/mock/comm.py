"""Communication context recorded during mock ``forward``."""

from __future__ import annotations

from contextvars import ContextVar, Token

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    CommCollective,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    CommOp,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.tensor import (
    Tensor,
)

_COMM: ContextVar[CommContext | None] = ContextVar(
    "hybridsim_mock_comm", default=None
)


class CommContext:
    """Insert COMM ops when ``num_ranks > 1``; otherwise pass the tensor through."""

    def __init__(self) -> None:
        self._token: Token[CommContext | None] | None = None

    def __enter__(self) -> CommContext:
        self._token = _COMM.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _COMM.reset(self._token)
            self._token = None

    def _emit(
        self,
        x: Tensor,
        *,
        name: str,
        collective: CommCollective,
        num_ranks: int,
    ) -> Tensor:
        ranks = max(1, int(num_ranks))
        if ranks <= 1:
            return x
        return CommOp.apply(
            x, name=name, collective=collective, num_ranks=ranks
        )

    def allreduce(self, x: Tensor, *, name: str, num_ranks: int) -> Tensor:
        return self._emit(
            x, name=name, collective=CommCollective.ALLREDUCE, num_ranks=num_ranks
        )

    def reduce_scatter(self, x: Tensor, *, name: str, num_ranks: int) -> Tensor:
        return self._emit(
            x,
            name=name,
            collective=CommCollective.REDUCE_SCATTER,
            num_ranks=num_ranks,
        )

    def allgather(self, x: Tensor, *, name: str, num_ranks: int) -> Tensor:
        return self._emit(
            x, name=name, collective=CommCollective.ALLGATHER, num_ranks=num_ranks
        )

    def dispatch(self, x: Tensor, *, name: str, num_ranks: int) -> Tensor:
        return self._emit(
            x, name=name, collective=CommCollective.DISPATCH, num_ranks=num_ranks
        )

    def combine(self, x: Tensor, *, name: str, num_ranks: int) -> Tensor:
        return self._emit(
            x, name=name, collective=CommCollective.COMBINE, num_ranks=num_ranks
        )

    def p2p(self, x: Tensor, *, name: str, num_ranks: int = 2) -> Tensor:
        return self._emit(
            x, name=name, collective=CommCollective.P2P, num_ranks=num_ranks
        )


def current_comm() -> CommContext:
    ctx = _COMM.get()
    if ctx is None:
        raise RuntimeError(
            "CommContext is not active; wrap forward in `with CommContext()`"
        )
    return ctx


class _CommProxy:
    """Module-level ``comm_ctx`` matching the design doc."""

    def allreduce(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().allreduce(x, **kwargs)  # type: ignore[arg-type]

    def reduce_scatter(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().reduce_scatter(x, **kwargs)  # type: ignore[arg-type]

    def allgather(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().allgather(x, **kwargs)  # type: ignore[arg-type]

    def dispatch(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().dispatch(x, **kwargs)  # type: ignore[arg-type]

    def combine(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().combine(x, **kwargs)  # type: ignore[arg-type]

    def p2p(self, x: Tensor, **kwargs: object) -> Tensor:
        return current_comm().p2p(x, **kwargs)  # type: ignore[arg-type]


comm_ctx = _CommProxy()
