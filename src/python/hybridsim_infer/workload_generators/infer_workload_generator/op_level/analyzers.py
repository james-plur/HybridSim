"""Named compute / comm analyzers for op-level lowering."""

from __future__ import annotations

from typing import Optional

from hybridsim_infer.workload_generators.configs import OpLevelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analyzer import (
    OpAnalyzer,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    AnalyticAnalyzer,
)

_COMPUTE_NAMES = frozenset({"analytic", "analytical"})
_COMM_ANALYTIC = frozenset({"analytic", "analytical", "ab", "alpha_beta", "auto", ""})
_COMM_RING = frozenset({"ring", "ring_comm", "comm"})


def make_compute_analyzer(
    name: str = "analytic",
    *,
    op_level: OpLevelConfig | None = None,
) -> OpAnalyzer:
    key = (name or "analytic").lower().strip()
    cfg = op_level or OpLevelConfig()
    if key in _COMPUTE_NAMES:
        return AnalyticAnalyzer(
            device=cfg.device,
            network=cfg.network,
            duration_scale=cfg.duration_scale,
        )
    raise ValueError(
        f"unknown compute analyzer {name!r}; expected one of {sorted(_COMPUTE_NAMES)}"
    )


def make_comm_analyzer(
    name: str = "analytic",
    *,
    replica_id: int = 0,
    num_ranks: int = 1,
    qos: int = 0,
) -> Optional[OpAnalyzer]:
    """Return a comm-only analyzer, or ``None`` to let compute lower CommOp."""
    key = (name or "analytic").lower().strip()
    if key in _COMM_ANALYTIC:
        return None
    if key in _COMM_RING:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.comm import (
            RingCommAnalyzer,
        )

        return RingCommAnalyzer(
            replica_id=int(replica_id),
            num_ranks=max(1, int(num_ranks)),
            qos=int(qos),
        )
    raise ValueError(
        f"unknown comm analyzer {name!r}; expected 'analytic' or 'ring'"
    )


def resolve_comm_analyzer_name(
    name: str,
    *,
    network_enabled: bool,
) -> str:
    key = (name or "analytic").lower().strip() or "analytic"
    if network_enabled and key in _COMM_ANALYTIC:
        return "ring"
    return key
