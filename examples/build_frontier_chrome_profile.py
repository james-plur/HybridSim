#!/usr/bin/env python3
"""Build a Chrome tracing profile from Frontier simulation outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _us(seconds: float) -> float:
    return float(seconds) * 1e6


def build_profile(run_dir: Path) -> dict:
    chrome_path = run_dir / "chrome_trace.json"
    ledger_path = run_dir / "frontier_stage_batch_ledger.jsonl"
    system_path = run_dir / "system_metrics.json"
    event_path = run_dir / "event_trace.json"

    with chrome_path.open(encoding="utf-8") as handle:
        base = json.load(handle)

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

    events: list[dict] = []

    # Metadata for chrome://tracing lanes.
    events.extend(
        [
            {
                "name": "process_name",
                "ph": "M",
                "pid": 0,
                "tid": 0,
                "args": {"name": "replica_0 / stage_0 (MONOLITHIC)"},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": 0,
                "args": {"name": "GPU batch execution"},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": 1,
                "args": {"name": "Scheduler events"},
            },
        ]
    )

    ledger_by_batch = {int(row["batch_id"]): row for row in ledger_rows}

    for trace_event in base.get("traceEvents", []):
        batch_id = int(trace_event.get("args", {}).get("batch_id", -1))
        ledger = ledger_by_batch.get(batch_id, {})
        tokens = trace_event.get("args", {}).get("num_tokens", [])
        phase = "prefill" if tokens and max(tokens) > 1 else "decode"
        total_ms = float(ledger.get("execution_time", {}).get("total_time_ms", 0.0))
        model_ms = float(ledger.get("execution_time", {}).get("model_time_ms", 0.0))

        events.append(
            {
                "name": f"{phase} batch_{batch_id}",
                "cat": "batch_execution",
                "ph": "X",
                "ts": trace_event["ts"],
                "dur": trace_event["dur"],
                "pid": trace_event.get("pid", 0),
                "tid": trace_event.get("tid", 0),
                "args": {
                    **trace_event.get("args", {}),
                    "phase": phase,
                    "total_time_ms": total_ms,
                    "model_time_ms": model_ms,
                    "stage_start_s": ledger.get("stage_start_ts"),
                    "stage_end_s": ledger.get("stage_end_ts"),
                },
            }
        )

        # Operator component slices inside the batch window.
        components = ledger.get("execution_time", {}).get("component_ledger_ms", {})
        if components and trace_event["dur"] > 0:
            start_us = float(trace_event["ts"])
            total_component_ms = sum(float(v) for v in components.values() if float(v) > 0)
            cursor = start_us
            scale = float(trace_event["dur"]) / (total_component_ms * 1000.0) if total_component_ms else 1.0
            for component_name, component_ms in sorted(
                components.items(),
                key=lambda item: item[0],
            ):
                component_ms = float(component_ms)
                if component_ms <= 0:
                    continue
                component_us = component_ms * 1000.0 * scale
                events.append(
                    {
                        "name": component_name,
                        "cat": "operator",
                        "ph": "X",
                        "ts": cursor,
                        "dur": component_us,
                        "pid": trace_event.get("pid", 0),
                        "tid": trace_event.get("tid", 0),
                        "args": {
                            "batch_id": batch_id,
                            "component_ms": component_ms,
                        },
                    }
                )
                cursor += component_us

    # Scheduler instant markers on a separate lane.
    event_type_names = {
        "global_schedule": "GlobalSchedule",
        "cluster_schedule": "ClusterSchedule",
        3: "ReplicaSchedule",
        12: "BatchStageArrival",
        11: "ReplicaStageSchedule",
        4: "BatchStageEnd",
        21: "ClusterBatchEnd",
        22: "GlobalBatchEnd",
    }
    for raw in event_trace:
        event_type = raw.get("event_type")
        label = event_type_names.get(event_type, str(event_type))
        time_s = float(raw.get("time", 0.0))
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

    profile = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": "Frontier discrete-event simulator",
            "run_dir": str(run_dir),
            "simulation_summary": system_metrics.get("simulation_metadata", {}),
            "throughput_metrics": system_metrics.get("throughput_metrics", {}),
            "ttft_statistics_ms": system_metrics.get("ttft_statistics", {}),
            "tpot_statistics_ms": system_metrics.get("tpot_statistics", {}),
            "modeling_notes": [
                "Frontier uses a priority-queue discrete-event simulation (DES).",
                "Requests are batched by vLLM v1 scheduler logic inside ReplicaScheduler.",
                "Each batch stage execution time is predicted by ExecutionTimePredictor.",
                "BatchStageEndEvent advances simulation clock and triggers next schedule.",
                "Chrome trace X-events map to batch stage GPU execution windows.",
            ],
        },
    }
    return profile


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path(
            "/home/y_luchenda/hybridsim/outputs/frontier_profile/"
            "meta_llama_llama_2_7b_hf/offline_batch/dense_basic_trace"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <run_dir>/inference_profile.json)",
    )
    args = parser.parse_args()

    output = args.output or (args.run_dir / "inference_profile.json")
    profile = build_profile(args.run_dir)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote Chrome tracing profile: {output}")
    print(f"Open chrome://tracing and load this file.")


if __name__ == "__main__":
    main()
