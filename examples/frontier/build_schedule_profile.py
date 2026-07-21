#!/usr/bin/env python3
"""Build Chrome tracing profiles from Frontier or hybridsim run outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _us(seconds: float) -> float:
    return float(seconds) * 1e6


def build_frontier_profile(run_dir: Path) -> dict:
    chrome_path = run_dir / "chrome_trace.json"
    ledger_path = run_dir / "frontier_stage_batch_ledger.jsonl"
    system_path = run_dir / "system_metrics.json"
    event_path = run_dir / "event_trace.json"

    events: list[dict] = []
    if chrome_path.exists():
        with chrome_path.open(encoding="utf-8") as handle:
            base = json.load(handle)
        events.extend(base.get("traceEvents", []))

    ledger_rows = []
    if ledger_path.exists():
        with ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ledger_rows.append(json.loads(line))

    system_metrics = {}
    if system_path.exists():
        with system_path.open(encoding="utf-8") as handle:
            system_metrics = json.load(handle)

    event_trace = []
    if event_path.exists():
        with event_path.open(encoding="utf-8") as handle:
            event_trace = json.load(handle)

    event_type_names = {
        "global_schedule": "GlobalSchedule",
        "cluster_schedule": "ClusterSchedule",
        3: "ReplicaSchedule",
        12: "BatchStageArrival",
        11: "ReplicaStageSchedule",
        4: "BatchStageEnd",
        21: "ClusterBatchEnd",
        22: "GlobalBatchEnd",
        23: "KV_CACHE_TRANSFER_START",
        24: "KV_CACHE_TRANSFER_END",
    }

    request_schedule: dict[int, list[dict]] = {}
    for raw in event_trace:
        event_type = raw.get("event_type")
        label = event_type_names.get(event_type, str(event_type))
        time_s = float(raw.get("time", 0.0))
        request_id = raw.get("request_id")
        entry = {
            "event": label,
            "time_s": time_s,
            "args": raw,
        }
        if request_id is not None:
            request_schedule.setdefault(int(request_id), []).append(entry)
        events.append(
            {
                "name": label,
                "cat": "scheduler",
                "ph": "i",
                "s": "t",
                "ts": _us(time_s),
                "pid": 0,
                "tid": 1,
                "args": raw,
            }
        )
        if label in {"KV_CACHE_TRANSFER_START", "KV_CACHE_TRANSFER_END"}:
            transfer_ms = float(raw.get("transfer_time_ms", 0.0))
            duration_s = transfer_ms * 1e-3 if transfer_ms else 0.0
            entry["duration_s"] = duration_s
            entry["event"] = "KVCacheTransfer"

    for row in ledger_rows:
        batch_id = int(row.get("batch_id", -1))
        request_ids = row.get("request_ids", [])
        start_s = float(row.get("stage_start_ts", 0.0))
        end_s = float(row.get("stage_end_ts", start_s))
        duration_s = max(0.0, end_s - start_s)
        entry = {
            "event": f"ledger_batch_{batch_id}",
            "time_s": start_s,
            "duration_s": duration_s,
            "args": row,
        }
        for request_id in request_ids:
            request_schedule.setdefault(int(request_id), []).append(entry)

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": "Frontier",
            "run_dir": str(run_dir),
            "simulation_summary": system_metrics.get("simulation_metadata", {}),
            "request_schedule": request_schedule,
            "ledger_rows": len(ledger_rows),
        },
    }


def build_hybridsim_profile(run_dir: Path) -> dict:
    profile_path = run_dir / "inference_profile.json"
    if profile_path.exists():
        with profile_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"Missing hybridsim profile: {profile_path}")


def write_profile(run_dir: Path, source: str) -> Path:
    if source == "frontier":
        profile = build_frontier_profile(run_dir)
    else:
        profile = build_hybridsim_profile(run_dir)
    output = run_dir / "inference_profile.json"
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return output


def _extract_batch_windows(profile: dict) -> list[tuple[int, float, float]]:
    seen: set[tuple[int, float, float]] = set()
    windows: list[tuple[int, float, float]] = []
    request_schedule = profile.get("metadata", {}).get("request_schedule", {})
    for events in request_schedule.values():
        for entry in events:
            name = entry.get("event", "")
            if not (
                name.startswith("batch_")
                or name.startswith("ledger_batch_")
                or name == "KVCacheTransfer"
            ):
                continue
            batch_id = entry.get("args", {}).get("batch_id")
            if batch_id is None and name.startswith("batch_"):
                batch_id = int(name.split("_", 1)[1])
            if batch_id is None:
                batch_id = len(windows)
            key = (
                int(batch_id),
                float(entry["time_s"]),
                float(entry.get("duration_s", 0.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            windows.append(key)
    return sorted(windows, key=lambda item: (item[0], item[1]))


def compare_profiles(frontier_profile: Path, hybridsim_profile: Path) -> dict:
    with frontier_profile.open(encoding="utf-8") as handle:
        frontier = json.load(handle)
    with hybridsim_profile.open(encoding="utf-8") as handle:
        hybridsim = json.load(handle)

    frontier_batches = _extract_batch_windows(frontier)
    hybridsim_batches = _extract_batch_windows(hybridsim)

    batch_mismatches = 0
    batch_diffs: list[dict] = []
    for index, (left, right) in enumerate(zip(frontier_batches, hybridsim_batches)):
        if abs(left[1] - right[1]) > 1e-3 or abs(left[2] - right[2]) > 1e-3:
            batch_mismatches += 1
            if len(batch_diffs) < 5:
                batch_diffs.append(
                    {
                        "index": index,
                        "frontier": {
                            "batch_id": left[0],
                            "start_s": left[1],
                            "duration_s": left[2],
                        },
                        "hybridsim": {
                            "batch_id": right[0],
                            "start_s": right[1],
                            "duration_s": right[2],
                        },
                    }
                )

    if len(frontier_batches) != len(hybridsim_batches):
        batch_mismatches += abs(len(frontier_batches) - len(hybridsim_batches))

    f_sched = frontier.get("metadata", {}).get("request_schedule", {})
    h_sched = hybridsim.get("metadata", {}).get("request_schedule", {})
    all_request_ids = sorted({int(k) for k in f_sched} | {int(k) for k in h_sched})

    return {
        "requests_compared": len(all_request_ids),
        "batch_windows_compared": min(len(frontier_batches), len(hybridsim_batches)),
        "frontier_batch_count": len(frontier_batches),
        "hybridsim_batch_count": len(hybridsim_batches),
        "batch_mismatches": batch_mismatches,
        "batch_match_rate": 0.0
        if not frontier_batches
        else (len(frontier_batches) - batch_mismatches) / len(frontier_batches),
        "mismatches": batch_mismatches,
        "match_rate": 0.0
        if not frontier_batches
        else (len(frontier_batches) - batch_mismatches) / len(frontier_batches),
        "batch_diff_samples": batch_diffs,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("run_dir", type=Path)
    build_p.add_argument("--source", choices=["frontier", "hybridsim"], required=True)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("frontier_profile", type=Path)
    cmp_p.add_argument("hybridsim_profile", type=Path)
    cmp_p.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "build":
        output = write_profile(args.run_dir, args.source)
        print(f"Wrote profile: {output}")
    else:
        result = compare_profiles(args.frontier_profile, args.hybridsim_profile)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote comparison: {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    main()
