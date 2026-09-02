"""Profile writer subprocess: drain Queue → Chrome Trace JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybridsim.request_profile.events import (
    MSG_COMPLETE,
    MSG_FLOW,
    MSG_INSTANT,
    MSG_PROFILE_META,
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


def _ensure_thread(
    events: list[dict[str, Any]],
    seen: set[str],
    pid: int,
    tid: int,
    name: str,
) -> None:
    key = f"thread:{int(pid)}:{int(tid)}"
    if key in seen:
        return
    seen.add(key)
    events.append(thread_name_event(int(pid), int(tid), str(name)))


def _ensure_cluster_meta(events: list[dict[str, Any]], seen: set[str]) -> None:
    key = "cluster"
    if key in seen:
        return
    seen.add(key)
    events.append(process_name_event(PID_CLUSTER, "Cluster"))
    _ensure_thread(events, seen, PID_CLUSTER, TID_CLUSTER_SCHEDULE, "schedule")
    _ensure_thread(events, seen, PID_CLUSTER, TID_CLUSTER_DISPATCH, "dispatch")


def _ensure_replica_meta(
    events: list[dict[str, Any]],
    seen: set[str],
    replica_id: int,
    *,
    process_name: str | None = None,
) -> None:
    key = f"replica_{int(replica_id)}"
    pid = replica_pid(replica_id)
    if key not in seen:
        seen.add(key)
        label = process_name or f"Replica_{int(replica_id)}"
        events.append(process_name_event(pid, str(label)))
        _ensure_thread(events, seen, pid, TID_REPLICA_ENGINE, "engine")
        _ensure_thread(events, seen, pid, TID_REPLICA_SCHEDULE, "schedule")


def _ensure_meta_for_msg(
    events: list[dict[str, Any]], seen: set[str], msg: dict[str, Any]
) -> None:
    replica_id = msg.get("replica_id")
    if replica_id is None and msg.get("track") == "cluster":
        _ensure_cluster_meta(events, seen)
    elif replica_id is not None:
        _ensure_replica_meta(
            events,
            seen,
            int(replica_id),
            process_name=msg.get("process_name"),
        )
    else:
        _ensure_cluster_meta(events, seen)
    thread_name = msg.get("thread_name")
    if thread_name is not None and replica_id is not None:
        _ensure_thread(
            events,
            seen,
            replica_pid(int(replica_id)),
            int(msg.get("tid", TID_REPLICA_ENGINE)),
            str(thread_name),
        )


def writer_main(queue: Any, output_path: str) -> None:
    """Child-process entry: consume events until STOP, then write JSON."""
    events: list[dict[str, Any]] = []
    seen_meta: set[str] = set()
    requests: dict[str, Any] = {}
    extra_meta: dict[str, Any] = {}
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
        if kind == MSG_PROFILE_META:
            payload = dict(msg.get("meta") or {})
            payload.pop("requests", None)
            extra_meta.update(payload)
            continue
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
    metadata: dict[str, Any] = {
        "source": "hybridsim_request_profile",
        "dropped_messages": dropped,
        "requests": requests,
    }
    metadata.update(extra_meta)
    profile = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": metadata,
    }
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
