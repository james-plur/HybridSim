"""Export op-level Operator DAG workloads as Chrome Trace (torch-profiler style)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from hybridsim_infer.schedule_types import ScheduleBatch
from hybridsim_infer.workload_generators.configs import OpLevelConfig, ParallelConfig
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (
    extract_batch_features,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    critical_path_duration_s,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.lower import (
    lower_op,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    OperatorKind,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.generator import (
    OpLevelWorkloadGenerator,
)

PathLike = Union[str, Path]

TID_COMPUTE = 0
TID_COMM = 1
TID_MEMORY = 2


def _tid_for_kernel(name: str, kind: Optional[str] = None) -> int:
    lower = (name or "").lower()
    if kind in (OperatorKind.COMM.value, "comm"):
        return TID_COMM
    if kind in (OperatorKind.MEM.value, "mem", "memory"):
        return TID_MEMORY
    if any(
        tok in lower
        for tok in (
            "allreduce",
            "allgather",
            "reduce_scatter",
            "send_recv",
            "dispatch",
            "combine",
            "p2p",
        )
    ):
        return TID_COMM
    if any(
        tok in lower
        for tok in (
            "layernorm",
            "residual",
            "emb",
            "kv_cache_save",
            "rope",
            "mlp_act",
            "share_act",
            "moe_topk",
            "moe_shuffle",
            "indexer",
        )
    ):
        return TID_MEMORY
    return TID_COMPUTE


def _us(seconds: float) -> float:
    return float(seconds) * 1e6


@dataclass
class ScheduledKernel:
    index: int
    name: str
    start_s: float
    duration_s: float
    dependencies: list[int]
    features: dict[str, Any]
    kind: str = "compute"

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


def asap_schedule(kernels: list[dict[str, Any]]) -> list[ScheduledKernel]:
    """Earliest-start schedule respecting TimeoutKernel dependencies."""
    n = len(kernels)
    starts = [0.0] * n
    scheduled: list[ScheduledKernel] = []
    for i, k in enumerate(kernels):
        deps = list(k.get("dependencies") or [])
        ready = max(
            (starts[d] + float(kernels[d].get("duration", 0.0)) for d in deps),
            default=0.0,
        )
        starts[i] = ready
        dur = float(k.get("duration", 0.0))
        feats = dict(k.get("features") or {})
        kind = str(k.get("kind") or feats.get("kind") or "")
        scheduled.append(
            ScheduledKernel(
                index=i,
                name=str(k.get("name", f"k{i}")),
                start_s=ready,
                duration_s=dur,
                dependencies=deps,
                features=feats,
                kind=kind,
            )
        )
    return scheduled


def _analyze_with_features(
    generator: OpLevelWorkloadGenerator,
    batch: ScheduleBatch,
    *,
    workload_id: int = 1,
) -> list[dict[str, Any]]:
    """Like AnalyticAnalyzer.analyze but keep flops/bytes/kind on each kernel."""
    op_dag = generator.build_dag(batch)
    analyzer = generator.analyzer
    kernels: list[dict[str, Any]] = []
    op_to_kernels: list[list[int]] = []

    for op_idx, op in enumerate(op_dag.operators):
        plan = lower_op(op)
        cross_deps: list[int] = []
        for dep_op in op.deps:
            pred = op_to_kernels[dep_op]
            cross_deps.append(pred[-1])
        uniq: list[int] = []
        seen: set[int] = set()
        for d in cross_deps:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        duration = analyzer.estimate_kernel_duration(plan)
        kind = plan.kind.value if hasattr(plan.kind, "value") else str(plan.kind)
        kernels.append(
            {
                "name": plan.name,
                "duration": float(duration),
                "dependencies": uniq,
                "features": dict(plan.features or {}),
                "kind": kind,
            }
        )
        op_to_kernels.append([len(kernels) - 1])
        _ = op_idx
    _ = workload_id
    return kernels


def rank_layout(parallel: ParallelConfig) -> list[tuple[int, int, int]]:
    """Return (global_rank, tp_rank, pp_stage) for each logical device."""
    tp = max(1, int(parallel.resolved_attn_tp()))
    pp = max(1, int(parallel.pp_size))
    ranks: list[tuple[int, int, int]] = []
    gid = 0
    for pp_stage in range(pp):
        for tp_rank in range(tp):
            ranks.append((gid, tp_rank, pp_stage))
            gid += 1
    return ranks


def build_chrome_trace(
    scheduled: list[ScheduledKernel],
    *,
    parallel: ParallelConfig | None = None,
    display_time_unit: str = "ns",
) -> dict[str, Any]:
    """Build a Chrome Trace JSON object from an ASAP schedule."""
    parallel = parallel or ParallelConfig()
    ranks = rank_layout(parallel)
    events: list[dict[str, Any]] = []
    tid_names = {
        TID_COMPUTE: "compute",
        TID_COMM: "comm",
        TID_MEMORY: "memory",
    }

    for gid, tp_rank, pp_stage in ranks:
        pid = gid + 1
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"rank{gid}_TP{tp_rank}_PP{pp_stage}"},
            }
        )
        for tid, tname in tid_names.items():
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": tname},
                }
            )

    for sk in scheduled:
        tid = _tid_for_kernel(sk.name, sk.kind)
        args = {
            "kernel_index": sk.index,
            "dependencies": sk.dependencies,
            "flops": sk.features.get("flops"),
            "bytes": sk.features.get("bytes"),
            "payload_bytes": sk.features.get("payload_bytes"),
        }
        for gid, _tp, _pp in ranks:
            events.append(
                {
                    "name": sk.name,
                    "cat": "op_dag",
                    "ph": "X",
                    "ts": _us(sk.start_s),
                    "dur": _us(max(0.0, sk.duration_s)),
                    "pid": gid + 1,
                    "tid": tid,
                    "args": args,
                }
            )

    return {
        "traceEvents": events,
        "displayTimeUnit": display_time_unit,
        "otherData": {
            "critical_path_s": critical_path_duration_s(
                [
                    {
                        "duration": sk.duration_s,
                        "dependencies": sk.dependencies,
                    }
                    for sk in scheduled
                ]
            ),
            "num_kernels": len(scheduled),
            "num_ranks": len(ranks),
        },
    }


def profile_schedule_batch(
    batch: ScheduleBatch,
    *,
    op_level: OpLevelConfig | None = None,
    model_preset: Optional[str] = None,
    workload_id: int = 1,
) -> dict[str, Any]:
    """Analyze ``batch`` and return a Chrome Trace dict."""
    from hybridsim_infer.workload_generators.model_config_resolve import (
        resolve_op_level_config,
    )

    cfg = resolve_op_level_config(
        op_level_config=op_level,
        model_preset=model_preset,
    )
    if cfg is None:
        cfg = OpLevelConfig()
    gen = OpLevelWorkloadGenerator(op_level=cfg)
    kernels = _analyze_with_features(gen, batch, workload_id=workload_id)
    scheduled = asap_schedule(kernels)
    return build_chrome_trace(scheduled, parallel=cfg.parallel)


def write_chrome_trace(trace: dict[str, Any], path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2)
    return out


def summarize_overlap(scheduled: Iterable[ScheduledKernel]) -> dict[str, Any]:
    """Simple stats: total serial sum vs critical path and compute/comm windows."""
    items = list(scheduled)
    if not items:
        return {"sum_s": 0.0, "critical_path_s": 0.0, "overlap_ratio": 0.0}
    sum_s = sum(sk.duration_s for sk in items)
    cp = critical_path_duration_s(
        [{"duration": sk.duration_s, "dependencies": sk.dependencies} for sk in items]
    )
    return {
        "sum_s": sum_s,
        "critical_path_s": cp,
        "overlap_ratio": (1.0 - cp / sum_s) if sum_s > 0 else 0.0,
        "num_kernels": len(items),
        "batch_features_hint": None,
    }


def make_demo_prefill_batch(
    *,
    chunk: int = 64,
    cached: int = 0,
    prompt: int | None = None,
) -> ScheduleBatch:
    """Helper for CLI / tests."""
    from hybridsim_infer.request import InferenceRequest
    from hybridsim_infer.schedule_types import PrefillChunk

    prompt_n = int(prompt if prompt is not None else max(chunk + cached, chunk))
    req = InferenceRequest(
        request_id=1,
        arrived_at=0.0,
        num_prefill_tokens=prompt_n,
        num_decode_tokens=8,
        num_computed_tokens=int(cached),
    )
    return ScheduleBatch(
        batch_id=1,
        requests=[req],
        tokens_per_request={1: int(chunk)},
        chunks=[PrefillChunk(request=req, num_tokens=int(chunk))],
    )


__all__ = [
    "ScheduledKernel",
    "asap_schedule",
    "build_chrome_trace",
    "extract_batch_features",
    "make_demo_prefill_batch",
    "profile_schedule_batch",
    "rank_layout",
    "summarize_overlap",
    "write_chrome_trace",
]
