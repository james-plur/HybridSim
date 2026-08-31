#!/usr/bin/env python3
"""Sweep KV Store pull bandwidth vs DRAM capacity (prefill-only, one replica).

Grid:
  bandwidth Gbps × store GB → mean TTFT, system TPS (prefill tokens / span),
  overall prefix hit rate (local APC ∪ Store).

Visit order (so early results already cover both axes):
  1. fix one store capacity, sweep all pull bandwidths
  2. fix one pull bandwidth, sweep remaining capacities
  3. fill the rest

Usage::

    PYTHONPATH=src/python:. python examples/inference/kv_store_pool_sweep.py \\
        --max-requests 200 --workers 4
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY = _REPO_ROOT / "src" / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from hybridsim_infer.workload_generators.kv_cache import (  # noqa: E402
    bytes_per_token,
)

MODEL_PRESET = "deepseek-v3.2"
TRACE_REL = Path("src/python/hybridsim_infer/request_generators/kvcache_traces/normalized/qwen_bailian_traceA.jsonl")
BLOCK_SIZE = 16
H100_HBM_GB = 80.0
H100_PEAK_FLOPS = 989e12 * 8  # 8× H100 BF16 peak
H100_HBM_BPS = 1e15  # inflated so compute is not HBM-bound
BANDWIDTHS_GBPS = (25, 50, 100, 200, 400)
STORE_CAPACITIES_GB = (32, 128, 512, 2048, 8192)
DEFAULT_BYTES_PER_TOKEN = 85888.0
PROGRESS_INTERVAL_S = 15.0
MEM_CHECK_INTERVAL_S = 5.0
MEM_LOW_AVAIL_GB = 24.0
MEM_CRITICAL_AVAIL_GB = 8.0
MEM_SHRINK_COOLDOWN_S = 20.0


def bytes_per_block(bpt: float, block_size: int = BLOCK_SIZE) -> float:
    return float(bpt) * int(block_size)


def store_blocks_for_gb(
    capacity_gb: float,
    *,
    bpt: float,
    block_size: int = BLOCK_SIZE,
) -> int:
    return int((float(capacity_gb) * 1e9) // bytes_per_block(bpt, block_size))


def gpu_num_blocks_for_hbm_gb(
    hbm_gb: float,
    *,
    bpt: float,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Include the reserved null block (never allocated)."""
    usable = store_blocks_for_gb(hbm_gb, bpt=bpt, block_size=block_size)
    return usable + 1


def _fmt_bw(bandwidth_gbps: float) -> str:
    """Stable label for float Gbps (3.125 → '3.125', 25.0 → '25')."""
    bw = float(bandwidth_gbps)
    if abs(bw - round(bw)) < 1e-9:
        return str(int(round(bw)))
    return f"{bw:g}"


def _cell_key(bandwidth_gbps: float, store_gb: float) -> str:
    return f"{_fmt_bw(bandwidth_gbps)}gbps_{int(store_gb)}gb"


def ordered_sweep_cells(
    bandwidths: list[float],
    capacities: list[int],
    *,
    anchor_capacity: int | None = None,
    anchor_bandwidth: float | None = None,
) -> list[tuple[float, int]]:
    """Visit cells so early results cover one full BW axis and one full capacity axis.

    1. Fix ``anchor_capacity`` (default: first capacity), sweep all bandwidths.
    2. Fix ``anchor_bandwidth`` (default: first bandwidth), sweep remaining capacities.
    3. Fill the rest, capacity-major then bandwidth.
    """
    bws = [float(x) for x in bandwidths]
    caps = [int(x) for x in capacities]
    if not bws or not caps:
        raise ValueError("bandwidths and capacities must be non-empty")
    cap0 = int(caps[0] if anchor_capacity is None else anchor_capacity)
    bw0 = float(bws[0] if anchor_bandwidth is None else anchor_bandwidth)
    if cap0 not in caps:
        raise ValueError(f"anchor_capacity={cap0} not in {caps}")
    if not any(abs(bw0 - b) < 1e-9 for b in bws):
        raise ValueError(f"anchor_bandwidth={bw0} not in {bws}")

    seen: set[str] = set()
    order: list[tuple[float, int]] = []

    def _add(bw: float, cap: int) -> None:
        key = _cell_key(bw, cap)
        if key not in seen:
            seen.add(key)
            order.append((float(bw), int(cap)))

    for bw in bws:
        _add(bw, cap0)
    for cap in caps:
        _add(bw0, cap)
    for cap in caps:
        for bw in bws:
            _add(bw, cap)
    return order


def _same_cell(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return abs(float(a.get("kv_bandwidth_gbps", -1)) - float(b.get("kv_bandwidth_gbps", -2))) < 1e-9 and int(
        a.get("kv_store_gb", -1)
    ) == int(b.get("kv_store_gb", -2))

def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"meta": {}, "cells": []}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"meta": {}, "cells": []}
    data = json.loads(raw)
    data.setdefault("meta", {})
    data.setdefault("cells", [])
    return data


def upsert_cell(path: Path, cell: dict[str, Any], meta: Optional[dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = load_results(path)
        if meta:
            data["meta"] = meta
        cells = [c for c in data.get("cells", []) if not _same_cell(c, cell)]
        cells.append(cell)
        cells.sort(key=lambda c: (int(c["kv_store_gb"]), float(c["kv_bandwidth_gbps"])))
        data["cells"] = cells
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def append_progress(path: Path, message: str) -> None:
    """Append one line to the shared progress log (and stdout)."""
    line = f"{_now_stamp()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_mem_available_gb() -> tuple[float, float]:
    """Return (MemAvailable GiB, MemTotal GiB) from /proc/meminfo."""
    avail_kb = 0.0
    total_kb = 0.0
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("MemAvailable:"):
                avail_kb = float(raw.split()[1])
            elif raw.startswith("MemTotal:"):
                total_kb = float(raw.split()[1])
    return avail_kb / (1024.0 * 1024.0), total_kb / (1024.0 * 1024.0)


def summarize_metrics(
    finished: list[Any],
    *,
    n_scheduled: int,
    sim_now: float,
) -> dict[str, Any]:
    if not finished:
        return {
            "mean_ttft_s": None,
            "tps": 0.0,
            "hit_rate": 0.0,
            "n_finished": 0,
            "n_scheduled": int(n_scheduled),
            "sim_now_s": float(sim_now),
            "prefill_tokens": 0,
            "prefix_hit_tokens": 0,
        }
    ttfts: list[float] = []
    for req in finished:
        finished_at = getattr(req, "finished_at", None)
        if finished_at is None:
            continue
        ttfts.append(float(finished_at) - float(req.arrived_at))
    prefill = sum(int(req.num_prefill_tokens) for req in finished)
    hits = sum(int(getattr(req, "prefix_hit_tokens", 0) or 0) for req in finished)
    t0 = min(float(req.arrived_at) for req in finished)
    t1 = max(
        float(getattr(req, "finished_at", None) or t0) for req in finished
    )
    span = max(t1 - t0, 1e-12)
    return {
        "mean_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "tps": float(prefill) / span,
        "hit_rate": (float(hits) / float(prefill)) if prefill else 0.0,
        "n_finished": len(finished),
        "n_scheduled": int(n_scheduled),
        "sim_now_s": float(sim_now),
        "prefill_tokens": int(prefill),
        "prefix_hit_tokens": int(hits),
    }


def run_cell(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: build one simulation from scalar args (do not pickle actors)."""
    try:
        from hybridsim_infer import InferenceConfig, build_inference_simulation
        from hybridsim_infer.request_generators import KvCacheTraceRequestGenerator
        from hybridsim_infer.workload_generators.configs import (
            DeviceConfig,
            OpLevelConfig,
            ParallelConfig,
        )
        from hybridsim_infer.workload_generators.kv_cache import (
            bytes_per_token as bpt_fn,
        )

        model_preset = str(payload.get("model_preset") or MODEL_PRESET)
        peak_flops = float(payload.get("peak_flops") or H100_PEAK_FLOPS)
        bpt = float(payload["bytes_per_token"])
        measured = float(bpt_fn(model_preset, num_tokens=1))
        if abs(measured - bpt) > 1e-6:
            raise AssertionError(
                f"KV bytes/token mismatch: preset={measured} expected={bpt}"
            )

        bandwidth_gbps = float(payload["kv_bandwidth_gbps"])
        store_gb = float(payload["kv_store_gb"])
        store_blocks = int(payload["kv_store_blocks"])
        gpu_blocks = int(payload["num_gpu_blocks"])
        max_requests = payload.get("max_requests")
        time_scale = float(payload.get("time_scale") or 1.0)
        trace_path = Path(payload["trace_path"])

        op_level = OpLevelConfig(
            device=DeviceConfig(
                peak_flops=peak_flops,
                hbm_bandwidth_bps=H100_HBM_BPS,
                compute_util=0.6,
                hbm_util=0.6,
            ),
            parallel=ParallelConfig(),
        )
        cfg = InferenceConfig(
            cluster_type="monolith",
            num_replicas=1,
            duration_mode="op_level",
            model_preset=model_preset,
            op_level_config=op_level,
            enable_kv_client=True,
            enable_prefix_caching=True,
            block_size=BLOCK_SIZE,
            store_block_size=BLOCK_SIZE,
            num_gpu_blocks=gpu_blocks,
            kv_store_blocks=store_blocks,
            kv_bandwidth_gbps=bandwidth_gbps,
            kv_latency_s=0.0,
            kv_transfer_s=1e-6,
            kv_lookup_async=False,
            kv_lookup_rtt_s=1e-4,
            tokens_per_step=8192,
            max_num_scheduled_tokens=8192,
            max_num_running_reqs=16,
            step_interval=1e-4,
            reserve_full_isl=True,
            enable_request_profile=False,
        )
        infra = build_inference_simulation(cfg)
        gen = KvCacheTraceRequestGenerator(
            trace_path,
            block_size=BLOCK_SIZE,
            max_requests=None if max_requests is None else int(max_requests),
            time_scale=time_scale,
        )
        requests = gen.generate()
        for req in requests:
            req.num_decode_tokens = 0
        infra.schedule_arrivals(requests)
        label = _cell_key(bandwidth_gbps, store_gb)
        progress_path = Path(payload["progress_path"])
        n_total = len(requests)
        stop = Event()

        def _report() -> None:
            n_fin = len(infra.finished_requests)
            append_progress(progress_path, f"[{label}] 请求：{n_fin}/{n_total}")

        def _progress_loop() -> None:
            last = -1
            while not stop.wait(float(payload.get("progress_interval_s") or PROGRESS_INTERVAL_S)):
                n_fin = len(infra.finished_requests)
                if n_fin != last:
                    last = n_fin
                    _report()

        append_progress(progress_path, f"[{label}] 请求：0/{n_total} 开始")
        reporter = Thread(target=_progress_loop, name=f"progress-{label}", daemon=True)
        reporter.start()
        try:
            infra.run()
        finally:
            stop.set()
            reporter.join(timeout=2.0)
            _report()
        infra.check_errors()
        metrics = summarize_metrics(
            list(infra.finished_requests),
            n_scheduled=len(requests),
            sim_now=float(infra.now),
        )
        cell = {
            "kv_bandwidth_gbps": float(bandwidth_gbps),
            "kv_store_gb": int(store_gb),
            "kv_store_blocks": store_blocks,
            "time_scale": time_scale,
            "ok": True,
            **metrics,
        }
        return cell
    except Exception as exc:  # noqa: BLE001 — worker must always return
        return {
            "kv_bandwidth_gbps": float(payload.get("kv_bandwidth_gbps", -1)),
            "kv_store_gb": int(payload.get("kv_store_gb", -1)),
            "kv_store_blocks": int(payload.get("kv_store_blocks", -1)),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "mean_ttft_s": None,
            "tps": None,
            "hit_rate": None,
        }


def _worker_entry(payload: dict[str, Any], result_q: Queue) -> None:
    """Child process: run one cell and always put a result unless SIGKILLed."""

    def _on_term(_signum: int, _frame: Any) -> None:
        result_q.put(
            {
                "kv_bandwidth_gbps": float(payload.get("kv_bandwidth_gbps", -1)),
                "kv_store_gb": int(payload.get("kv_store_gb", -1)),
                "ok": False,
                "retry": True,
                "error": "terminated_for_memory",
            }
        )
        os._exit(15)

    signal.signal(signal.SIGTERM, _on_term)
    result_q.put(run_cell(payload))


def _terminate_worker(proc: Process, job: dict[str, Any], progress_path: Path) -> None:
    label = _cell_key(job["kv_bandwidth_gbps"], job["kv_store_gb"])
    append_progress(progress_path, f"[mem] 终止 {label} pid={proc.pid} 以降低内存")
    if proc.is_alive():
        proc.terminate()


def run_dynamic_pool(
    jobs: list[dict[str, Any]],
    *,
    max_workers: int,
    output: Path,
    meta: dict[str, Any],
    progress_path: Path,
    mem_low_gb: float,
    mem_critical_gb: float,
) -> int:
    """Run jobs with a shrink-only worker cap driven by MemAvailable."""
    pending = list(jobs)
    running: list[tuple[Process, dict[str, Any], Queue]] = []
    stopping: list[tuple[Process, dict[str, Any], Queue]] = []
    target = max(1, int(max_workers))
    last_shrink = 0.0
    last_mem_log = 0.0
    failed = 0
    finished_ok = 0

    append_progress(
        progress_path,
        f"[pool] 启动 pending={len(pending)} workers={target} "
        f"mem_low={mem_low_gb:.1f}GB mem_critical={mem_critical_gb:.1f}GB",
    )

    while pending or running or stopping:
        def _reap(
            bucket: list[tuple[Process, dict[str, Any], Queue]],
        ) -> list[tuple[Process, dict[str, Any], Queue]]:
            nonlocal failed, finished_ok
            alive: list[tuple[Process, dict[str, Any], Queue]] = []
            for proc, job, q in bucket:
                if proc.is_alive():
                    alive.append((proc, job, q))
                    continue
                proc.join(timeout=1.0)
                cell: dict[str, Any] | None = None
                try:
                    cell = q.get_nowait()
                except Empty:
                    cell = None
                label = _cell_key(job["kv_bandwidth_gbps"], job["kv_store_gb"])
                if cell is None or cell.get("retry"):
                    pending.insert(0, job)
                    append_progress(progress_path, f"[{label}] 被中断，重新排队")
                    continue
                upsert_cell(output, cell, meta=meta)
                if cell.get("ok"):
                    finished_ok += 1
                    append_progress(
                        progress_path,
                        f"[{label}] 完成 请求：{cell.get('n_finished')}/{cell.get('n_scheduled')} "
                        f"ttft={cell.get('mean_ttft_s')} tps={cell.get('tps')} hit={cell.get('hit_rate')}",
                    )
                else:
                    failed += 1
                    append_progress(
                        progress_path,
                        f"[{label}] 失败 {cell.get('error')}",
                    )
                    if cell.get("traceback"):
                        print(cell["traceback"], flush=True)
            return alive

        running = _reap(running)
        stopping = _reap(stopping)

        avail_gb, total_gb = read_mem_available_gb()
        now = time.time()
        if now - last_mem_log >= 30.0:
            append_progress(
                progress_path,
                f"[mem] avail={avail_gb:.1f}GB/{total_gb:.1f}GB "
                f"workers={len(running)+len(stopping)}/{target} pending={len(pending)} "
                f"done={finished_ok} fail={failed}",
            )
            last_mem_log = now

        new_target = target
        if avail_gb < mem_critical_gb:
            new_target = 1
        elif avail_gb < mem_low_gb:
            new_target = max(1, target - 1)
        if new_target < target and now - last_shrink >= MEM_SHRINK_COOLDOWN_S:
            append_progress(
                progress_path,
                f"[mem] 可用内存 {avail_gb:.1f}GB < 阈值，进程数 {target} → {new_target}",
            )
            target = new_target
            last_shrink = now
            while len(running) > target:
                proc, job, q = running.pop()
                _terminate_worker(proc, job, progress_path)
                stopping.append((proc, job, q))

        while pending and (len(running) + len(stopping)) < target:
            job = pending.pop(0)
            q: Queue = Queue()
            proc = Process(target=_worker_entry, args=(job, q), daemon=False)
            proc.start()
            running.append((proc, job, q))
            label = _cell_key(job["kv_bandwidth_gbps"], job["kv_store_gb"])
            append_progress(progress_path, f"[{label}] 启动 pid={proc.pid}")

        time.sleep(MEM_CHECK_INTERVAL_S)

    return failed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    default_out = (
        _REPO_ROOT / "examples" / "inference" / "results" / "kv_store_pool_sweep.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0, help="0 → min(8, cpu_count)")
    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=list(BANDWIDTHS_GBPS),
    )
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=list(STORE_CAPACITIES_GB),
    )
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--trace", type=Path, default=_REPO_ROOT / TRACE_REL)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run cells already present in --output",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=_REPO_ROOT / "examples" / "inference" / "results" / "sweep_progress.log",
        help="Per-experiment request progress log (请求：xx/xx)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_INTERVAL_S,
        help="Seconds between 请求：xx/xx lines",
    )
    parser.add_argument(
        "--mem-low-gb",
        type=float,
        default=MEM_LOW_AVAIL_GB,
        help="Shrink one worker when MemAvailable falls below this",
    )
    parser.add_argument(
        "--mem-critical-gb",
        type=float,
        default=MEM_CRITICAL_AVAIL_GB,
        help="Force workers=1 when MemAvailable falls below this",
    )
    parser.add_argument(
        "--anchor-capacity",
        type=int,
        default=None,
        help="Phase-1 fixed store GB (default: first --capacities value)",
    )
    parser.add_argument(
        "--anchor-bandwidth",
        type=float,
        default=None,
        help="Phase-2 fixed pull bandwidth Gbps (default: first --bandwidths value)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiply trace timestamps (e.g. 0.5 halves inter-arrival times)",
    )
    parser.add_argument(
        "--model-preset",
        type=str,
        default=MODEL_PRESET,
        help="Analytical model preset id",
    )
    parser.add_argument(
        "--hbm-gb",
        type=float,
        default=H100_HBM_GB,
        help="GPU HBM capacity used to size num_gpu_blocks",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=H100_PEAK_FLOPS,
        help="Device peak FLOPS (default: 8× H100 BF16)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    model_preset = str(args.model_preset)
    bpt = float(bytes_per_token(model_preset, num_tokens=1))
    if model_preset == MODEL_PRESET and abs(bpt - DEFAULT_BYTES_PER_TOKEN) > 1e-6:
        raise SystemExit(
            f"deepseek-v3.2 bytes/token={bpt}, expected {DEFAULT_BYTES_PER_TOKEN}"
        )
    gpu_blocks = gpu_num_blocks_for_hbm_gb(float(args.hbm_gb), bpt=bpt)
    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise SystemExit(f"trace not found: {trace_path}")

    meta = {
        "model_preset": model_preset,
        "bytes_per_token": bpt,
        "block_size": BLOCK_SIZE,
        "bytes_per_block": bytes_per_block(bpt),
        "num_gpu_blocks": gpu_blocks,
        "h100_hbm_gb": float(args.hbm_gb),
        "h100_peak_flops": float(args.peak_flops),
        "h100_hbm_bandwidth_bps": H100_HBM_BPS,
        "trace": str(trace_path),
        "time_scale": float(args.time_scale),
        "max_requests": args.max_requests,
        "cluster_type": "monolith",
        "num_replicas": 1,
        "prefill_only": True,
        "parallel": "tp=pp=ep=1",
        "sweep_order": "anchor_capacity_then_anchor_bandwidth_then_rest",
        "anchor_capacity_gb": int(
            args.capacities[0] if args.anchor_capacity is None else args.anchor_capacity
        ),
        "anchor_bandwidth_gbps": float(
            args.bandwidths[0] if args.anchor_bandwidth is None else args.anchor_bandwidth
        ),
    }
    existing = load_results(args.output)
    done: set[str] = set()
    if not args.no_skip_existing:
        for cell in existing.get("cells", []):
            if cell.get("ok"):
                done.add(_cell_key(cell["kv_bandwidth_gbps"], cell["kv_store_gb"]))

    jobs: list[dict[str, Any]] = []
    cell_order = ordered_sweep_cells(
        [float(x) for x in args.bandwidths],
        [int(x) for x in args.capacities],
        anchor_capacity=args.anchor_capacity,
        anchor_bandwidth=args.anchor_bandwidth,
    )
    meta["cell_order"] = [_cell_key(bw, cap) for bw, cap in cell_order]
    for bw, store_gb in cell_order:
        key = _cell_key(bw, store_gb)
        if key in done:
            continue
        jobs.append(
            {
                "kv_bandwidth_gbps": float(bw),
                "kv_store_gb": int(store_gb),
                "kv_store_blocks": store_blocks_for_gb(store_gb, bpt=bpt),
                "num_gpu_blocks": gpu_blocks,
                "bytes_per_token": bpt,
                "model_preset": model_preset,
                "peak_flops": float(args.peak_flops),
                "max_requests": args.max_requests,
                "time_scale": float(args.time_scale),
                "trace_path": str(trace_path),
                "progress_path": str(args.progress_log),
                "progress_interval_s": float(args.progress_interval),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_results(args.output)
    data["meta"] = meta
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    n_cpu = os.cpu_count() or 1
    workers = int(args.workers) or min(8, n_cpu, max(1, len(jobs)))
    workers = max(1, min(workers, max(1, len(jobs))))
    print(
        f"cells={len(jobs)} skip={len(done)} workers={workers} "
        f"model={model_preset} hbm_gb={args.hbm_gb} peak_flops={args.peak_flops} "
        f"time_scale={args.time_scale} bpt={bpt} gpu_blocks={gpu_blocks} "
        f"progress={args.progress_log} "
        f"order={meta['cell_order']}",
        flush=True,
    )
    if not jobs:
        print(f"nothing to run; results at {args.output}", flush=True)
        return 0

    failed = run_dynamic_pool(
        jobs,
        max_workers=workers,
        output=args.output,
        meta=meta,
        progress_path=args.progress_log,
        mem_low_gb=float(args.mem_low_gb),
        mem_critical_gb=float(args.mem_critical_gb),
    )
    print(f"wrote {args.output} failed={failed}/{len(jobs)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
