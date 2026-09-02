#!/usr/bin/env python3
"""Multi Prefill + Multi Decode PD demo with KV transfer, prefix cache, and profile.

Request input:
  --input handwritten  (default)  six shared-prefix prompts
  --input trace                   first N records of a KV cache JSONL (default 10)

Workload:
  --workload batch  (default)  token-proportional batch duration
  --workload op                mock op DAG + Roofline; profile shows compute/comm streams

Always prints ``metrics()``. Open the Chrome Trace JSON in chrome://tracing or Perfetto.

Examples::

    PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py
    PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py \\
        --input trace --workload op
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from hybridsim.request_profile import default_profile_dir
from hybridsim_infer import (
    ArtifactOutput,
    BatchLevelConfig,
    BatchTokenProportionalConfig,
    ClusterConfig,
    DeviceConfig,
    InferWorkloadConfig,
    InferenceConfig,
    InferenceRequest,
    KvConfig,
    KvLookupConfig,
    KvWorkloadConfig,
    ModelSpec,
    OpLevelConfig,
    OutputConfig,
    ParallelConfig,
    ReplicaScheduleConfig,
    RequestProfileOutput,
    ScheduleConfig,
    build_inference_simulation,
)
from hybridsim_infer.request_generators import (
    KVCACHE_TRACES_DIR,
    KvCacheTraceRequestGenerator,
)

DEFAULT_TRACE = KVCACHE_TRACES_DIR / "normalized" / "mooncake_fast25.jsonl"
SAMPLE_TRACE = KVCACHE_TRACES_DIR / "samples" / "mooncake_fast25.head.jsonl"


# A100-ish Llama-3.1-8B serving ballpark (not the package toy defaults).
# Prefill ~7k tok/s; decode ~50 tok/s (HBM-bound weight read + launch overhead).
_PREFILL_S_PER_TOKEN = 1.5e-4
_DECODE_S_PER_TOKEN = 2.0e-2
# Small-batch util is below datasheet; decode especially cannot hit 0.6 HBM.
_A100_DEVICE = DeviceConfig(
    peak_flops=312e12,
    hbm_bandwidth_bps=2.039e12,
    compute_util=0.40,
    hbm_util=0.35,
)


def _handwritten_requests() -> list[InferenceRequest]:
    # ~256-token prompts so prefill bars are tens of ms, not sub-ms spikes.
    shared_prefix = list(range(100, 100 + 128))
    suffix_a = list(range(200, 328))
    suffix_b = list(range(400, 528))
    suffix_c = list(range(600, 728))
    prompt_a = shared_prefix + suffix_a
    prompt_b = shared_prefix + suffix_b
    prompt_c = list(prompt_a)
    prompt_d = shared_prefix + suffix_c
    return [
        InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt_a),
            num_decode_tokens=16,
            prompt_token_ids=list(prompt_a),
        ),
        InferenceRequest(
            request_id=2,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt_b),
            num_decode_tokens=12,
            prompt_token_ids=list(prompt_b),
        ),
        InferenceRequest(
            request_id=3,
            arrived_at=0.002,
            num_prefill_tokens=len(prompt_d),
            num_decode_tokens=12,
            prompt_token_ids=list(prompt_d),
        ),
        InferenceRequest(
            request_id=4,
            arrived_at=0.002,
            num_prefill_tokens=len(prompt_b),
            num_decode_tokens=8,
            prompt_token_ids=list(prompt_b),
        ),
        # After first Prefills finish (~50–80ms) so local prefix cache can hit.
        InferenceRequest(
            request_id=5,
            arrived_at=0.12,
            num_prefill_tokens=len(prompt_c),
            num_decode_tokens=16,
            prompt_token_ids=list(prompt_c),
        ),
        InferenceRequest(
            request_id=6,
            arrived_at=0.13,
            num_prefill_tokens=len(prompt_a),
            num_decode_tokens=8,
            prompt_token_ids=list(prompt_a),
        ),
    ]


def _resolve_trace_path(path: Optional[Path]) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise SystemExit(f"trace not found: {resolved}")
        return resolved
    if DEFAULT_TRACE.exists():
        return DEFAULT_TRACE
    if SAMPLE_TRACE.exists():
        return SAMPLE_TRACE
    raise SystemExit(
        f"no KV trace found at {DEFAULT_TRACE} or {SAMPLE_TRACE}; pass --trace"
    )


def _load_trace_requests(
    *,
    trace_path: Path,
    max_requests: int,
    block_size: int,
    max_decode: Optional[int],
) -> list[InferenceRequest]:
    gen = KvCacheTraceRequestGenerator(
        trace_path,
        block_size=block_size,
        max_requests=max_requests,
    )
    requests = gen.generate()
    if max_decode is not None:
        for req in requests:
            req.num_decode_tokens = min(int(req.num_decode_tokens), int(max_decode))
    return requests


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        choices=("handwritten", "trace"),
        default="handwritten",
        help="request source",
    )
    parser.add_argument(
        "--workload",
        choices=("batch", "op"),
        default="batch",
        help="infer duration mode: batch_level or op_level",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="KV cache JSONL (default: mooncake_fast25 normalized, else sample head)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10,
        help="trace only: max requests (default 10)",
    )
    parser.add_argument(
        "--max-decode",
        type=int,
        default=-1,
                        help="cap decode tokens; -1 = auto (16 for op-level, unlimited for batch); 0 = unlimited",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="op-level tensor-parallel size (default 2 so CommOps appear)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="KV block size; 0 = 16 for handwritten, 512 for trace",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Chrome Trace JSON path",
    )
    parser.add_argument(
        "--write-metrics",
        action="store_true",
        help="also write metrics.json next to the profile",
    )
    return parser.parse_args(argv)


def _resolved_max_decode(args: argparse.Namespace) -> Optional[int]:
    if int(args.max_decode) == 0:
        return None
    if int(args.max_decode) > 0:
        return int(args.max_decode)
    if args.workload == "op":
        return 16
    return None


def _profile_path(args: argparse.Namespace) -> Path:
    if args.profile_path is not None:
        return Path(args.profile_path)
    if args.input == "handwritten" and args.workload == "batch":
        return default_profile_dir() / "pd_multipool_profile_demo.json"
    return default_profile_dir() / f"pd_multipool_profile_demo_{args.input}_{args.workload}.json"


def build_config(args: argparse.Namespace) -> InferenceConfig:
    is_trace = args.input == "trace"
    is_op = args.workload == "op"
    block_size = int(args.block_size) if int(args.block_size) > 0 else (512 if is_trace else 16)
    tokens_per_step = 2048 if is_trace else 64
    max_scheduled = 8192 if is_trace else 256
    num_gpu_blocks = 8192 if is_trace else 2048

    if is_op:
        infer_workload = InferWorkloadConfig(
            mode="op_level",
            op=OpLevelConfig(
                parallel=ParallelConfig(tp_size=max(1, int(args.tp))),
                device=_A100_DEVICE,
            ),
        )
    else:
        infer_workload = InferWorkloadConfig(
            mode="batch_level",
            batch=BatchLevelConfig(
                predictor="token_proportional",
                token_proportional=BatchTokenProportionalConfig(
                    prefill_s_per_token=_PREFILL_S_PER_TOKEN,
                    decode_s_per_token=_DECODE_S_PER_TOKEN,
                ),
            ),
        )

    metrics_path = None
    if args.write_metrics:
        metrics_path = _profile_path(args).with_name(
            _profile_path(args).stem + "_metrics.json"
        )

    return InferenceConfig(
        cluster=ClusterConfig(
            type="pd",
            num_prefill_replicas=2,
            num_decode_replicas=2,
        ),
        schedule=ScheduleConfig(
            replica=ReplicaScheduleConfig(
                tokens_per_step=tokens_per_step,
                decode_tokens_per_step=1,
                max_num_scheduled_tokens=max_scheduled,
                max_num_running_reqs=16,
            ),
        ),
        kv=KvConfig(
            enable_store=True,
            enable_prefix_caching=True,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            lookup=KvLookupConfig(rtt_s=1e-3),
        ),
        model=ModelSpec(preset="llama-3.1-8b"),
        infer_workload=infer_workload,
        kv_workload=KvWorkloadConfig(
            bandwidth_gbps=50.0,
            transfer_s_floor=1e-4,
        ),
        output=OutputConfig(
            request_profile=RequestProfileOutput(
                enabled=True,
                path=_profile_path(args),
            ),
            metrics=ArtifactOutput(
                enabled=bool(args.write_metrics),
                path=metrics_path,
            ),
        ),
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    max_decode = _resolved_max_decode(args)
    trace_path: Optional[Path] = None
    if args.input == "handwritten":
        requests = _handwritten_requests()
    else:
        block_size = int(args.block_size) if int(args.block_size) > 0 else 512
        trace_path = _resolve_trace_path(args.trace)
        requests = _load_trace_requests(
            trace_path=trace_path,
            max_requests=max(1, int(args.max_requests)),
            block_size=block_size,
            max_decode=max_decode,
        )
        if not requests:
            raise SystemExit(f"no requests loaded from {trace_path}")

    if args.input == "handwritten" and max_decode is not None:
        for req in requests:
            req.num_decode_tokens = min(int(req.num_decode_tokens), int(max_decode))

    cfg = build_config(args)
    infra = build_inference_simulation(cfg)
    assert len(infra.replicas) == 4

    emit_meta = getattr(infra.profile, "emit_profile_meta", None)
    if emit_meta is not None:
        emit_meta(
            {
                "input": args.input,
                "workload": args.workload,
                "tp": int(args.tp) if args.workload == "op" else 1,
                "n_requests": len(requests),
                "trace": str(trace_path) if trace_path is not None else None,
                "max_decode": max_decode,
                "model_preset": "llama-3.1-8b",
            }
        )

    print("=== PD multipool profile demo (2P+2D, KV, prefix) ===")
    print(
        f"input={args.input} workload={args.workload} "
        f"replicas={len(infra.replicas)} n_requests={len(requests)} "
        f"profile={cfg.output.request_profile.path}"
    )
    infra.schedule_arrivals(requests)
    infra.run()
    infra.check_errors()

    print("metrics:")
    print(json.dumps(infra.metrics(), indent=2, sort_keys=True))

    finished = sorted(infra.finished_requests, key=lambda r: r.request_id)
    print(
        f"arrived={infra.cluster.arrived_count} finished={len(finished)} "
        f"now={infra.now:.4f}s"
    )
    for req in finished:
        params = req.kv_transfer_params or {}
        print(
            f"  req={req.request_id} completed={req.completed} "
            f"src_P={params.get('remote_replica_id')} "
            f"handed_off={bool(params.get('_handed_off'))} "
            f"prefill={req.num_prefill_tokens} decode={req.num_decode_tokens}"
        )
        if not req.completed:
            raise SystemExit(f"req {req.request_id} did not complete")

    if len(finished) != len(requests):
        raise SystemExit(f"expected {len(requests)} finished, got {len(finished)}")
    if infra.profile_path is not None:
        print(f"request profile: {infra.profile_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")
    print("ok")


if __name__ == "__main__":
    main()
