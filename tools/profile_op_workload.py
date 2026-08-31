#!/usr/bin/env python3
"""CLI: export op-level DAG as Chrome Trace JSON (Perfetto / chrome://tracing).

Example:
  PYTHONPATH=src/python python tools/profile_op_workload.py \\
    --preset llama-3.1-8b --chunk 64 --cached 32 --tp 2 --pp 2 \\
    -o /tmp/op_dag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PY = _ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.workload_generators.configs import (  # noqa: E402
    OpLevelConfig,
    ParallelConfig,
)
from hybridsim_infer.workload_generators.infer_workload_generator.batch_features import (  # noqa: E402
    extract_batch_features,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.dag_profile import (  # noqa: E402
    asap_schedule,
    make_demo_prefill_batch,
    profile_schedule_batch,
    summarize_overlap,
    write_chrome_trace,
    _analyze_with_features,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.generator import (  # noqa: E402
    OpLevelWorkloadGenerator,
)
from hybridsim_infer.workload_generators.model_config_resolve import (  # noqa: E402
    resolve_op_level_config,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default=None, help="model_preset id, e.g. llama-3.1-8b")
    p.add_argument("--chunk", type=int, default=64, help="prefill tokens this step")
    p.add_argument("--cached", type=int, default=0, help="already-cached prefix tokens")
    p.add_argument("--prompt", type=int, default=None, help="full prompt length")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size (trace ranks)")
    p.add_argument("--pp", type=int, default=1, help="pipeline-parallel size (trace ranks)")
    p.add_argument("-o", "--output", default="op_dag_trace.json", help="Chrome Trace path")
    p.add_argument(
        "--summary",
        action="store_true",
        help="print overlap summary JSON to stdout",
    )
    args = p.parse_args(argv)

    batch = make_demo_prefill_batch(
        chunk=args.chunk, cached=args.cached, prompt=args.prompt
    )
    feats = extract_batch_features(batch)
    cfg = resolve_op_level_config(
        op_level_config=OpLevelConfig(
            parallel=ParallelConfig(tp_size=max(1, args.tp), pp_size=max(1, args.pp))
        ),
        model_preset=args.preset,
    )
    if cfg is not None:
        cfg.parallel = ParallelConfig(tp_size=max(1, args.tp), pp_size=max(1, args.pp))

    trace = profile_schedule_batch(batch, op_level=cfg, model_preset=None)
    out = write_chrome_trace(trace, args.output)

    gen = OpLevelWorkloadGenerator(op_level=cfg or OpLevelConfig())
    scheduled = asap_schedule(_analyze_with_features(gen, batch))
    summary = summarize_overlap(scheduled)
    summary["batch_features"] = {
        "phase": feats.phase.value,
        "num_prefill_tokens": feats.num_prefill_tokens,
        "cached_prefix_tokens": feats.cached_prefix_tokens,
        "cached_decode_tokens": feats.cached_decode_tokens,
    }
    summary["output"] = str(out)
    summary["num_ranks"] = int(trace.get("otherData", {}).get("num_ranks", 1))

    if args.summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"wrote {out} "
            f"(kernels={summary['num_kernels']} ranks={summary['num_ranks']} "
            f"cp={summary['critical_path_s']:.6e}s "
            f"overlap={summary['overlap_ratio']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
