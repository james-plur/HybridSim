"""Profile writer subprocess: drain Queue → Chrome Trace JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybridsim.request_profile.events import (
    MSG_COMPLETE,
    MSG_FLOW,
    MSG_INSTANT,
    MSG_REQUEST_META,
    MSG_STOP,
    PID_CLUSTER,
    TID_CLUSTER_DISPATCH,
    TID_CLUSTER_SCHEDULE,
    TID_REPLICA_ENGINE,
    TID_REPLICA_SCHEDULE,
    complete_event,
    flow_event,
    instant_event,
    process_name_event,
    replica_pid,
    thread_name_event,
)


def _ensure_cluster_meta(events: list[dict[str, Any]], seen: set[str]) -> None:
    key = "cluster"
    if key in seen:
        return
    seen.add(key)
    events.append(process_name_event(PID_CLUSTER, "Cluster"))
    events.append(
        thread_name_event(PID_CLUSTER, TID_CLUSTER_SCHEDULE, "schedule")
    )
    events.append(
        thread_name_event(PID_CLUSTER, TID_CLUSTER_DISPATCH, "dispatch")
    )


def _ensure_replica_meta(
    events: list[dict[str, Any]], seen: set[str], replica_id: int
) -> None:
    key = f"replica_{int(replica_id)}"
    if key in seen:
        return
    seen.add(key)
    pid = replica_pid(replica_id)
    events.append(process_name_event(pid, f"Replica_{int(replica_id)}"))
    events.append(thread_name_event(pid, TID_REPLICA_ENGINE, "engine"))
    events.append(thread_name_event(pid, TID_REPLICA_SCHEDULE, "schedule"))


def _ensure_meta_for_msg(
    events: list[dict[str, Any]], seen: set[str], msg: dict[str, Any]
) -> None:
    replica_id = msg.get("replica_id")
    if replica_id is None and msg.get("track") == "cluster":
        _ensure_cluster_meta(events, seen)
    elif replica_id is not None:
        _ensure_replica_meta(events, seen, int(replica_id))
    else:
        _ensure_cluster_meta(events, seen)


def writer_main(queue: Any, output_path: str) -> None:
    """Child-process entry: consume events until STOP, then write JSON."""
    events: list[dict[str, Any]] = []
    seen_meta: set[str] = set()
    requests: dict[str, Any] = {}
    dropped = 0

    while True:
        try:
            msg = queue.get()
        except (EOFError, OSError, KeyboardInterrupt):
            break
        if msg is None:
            break
        kind = msg.get("kind")
        if kind == MSG_STOP:
            break
        if kind == MSG_REQUEST_META:
            meta = dict(msg.get("meta") or {})
            rid = meta.get("request_id")
            if rid is None:
                dropped += 1
                continue
            key = str(int(rid))
            prev = dict(requests.get(key) or {})
            prev.update(meta)
            requests[key] = prev
        elif kind == MSG_COMPLETE:
            _ensure_meta_for_msg(events, seen_meta, msg)
            events.append(
                complete_event(
                    name=str(msg["name"]),
                    start_s=float(msg["start_s"]),
                    duration_s=float(msg.get("duration_s", 0.0)),
                    pid=int(msg["pid"]),
                    tid=int(msg["tid"]),
                    category=str(msg.get("category", "request_profile")),
                    args=msg.get("args"),
                )
            )
        elif kind == MSG_INSTANT:
            _ensure_meta_for_msg(events, seen_meta, msg)
            events.append(
                instant_event(
                    name=str(msg["name"]),
                    time_s=float(msg["time_s"]),
                    pid=int(msg["pid"]),
                    tid=int(msg["tid"]),
                    category=str(msg.get("category", "request_profile")),
                    args=msg.get("args"),
                )
            )
        elif kind == MSG_FLOW:
            _ensure_meta_for_msg(events, seen_meta, msg)
            events.append(
                flow_event(
                    name=str(msg["name"]),
                    phase=str(msg["phase"]),
                    time_s=float(msg["time_s"]),
                    pid=int(msg["pid"]),
                    tid=int(msg["tid"]),
                    flow_id=int(msg["flow_id"]),
                    category=str(msg.get("category", "flow")),
                    args=msg.get("args"),
                )
            )
        else:
            dropped += 1

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": "hybridsim_request_profile",
            "dropped_messages": dropped,
            "requests": requests,
        },
    }
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
