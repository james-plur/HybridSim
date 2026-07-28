"""Chrome Trace event helpers for request-level profiling."""

from __future__ import annotations

from typing import Any, Optional

# Queue message kinds (parent → writer process)
MSG_COMPLETE = "complete"
MSG_INSTANT = "instant"
MSG_FLOW = "flow"
MSG_REQUEST_META = "request_meta"
MSG_STOP = "stop"

# Fixed Chrome Trace pids / tids
PID_CLUSTER = 1
TID_CLUSTER_SCHEDULE = 0
TID_CLUSTER_DISPATCH = 1

TID_REPLICA_ENGINE = 0
TID_REPLICA_SCHEDULE = 1


def us(seconds: float) -> float:
    return float(seconds) * 1e6


def replica_pid(replica_id: int) -> int:
    """Stable pid for replica tracks (Cluster occupies pid=1)."""
    return int(replica_id) + 2


def process_name_event(pid: int, name: str) -> dict[str, Any]:
    return {
        "name": "process_name",
        "ph": "M",
        "pid": pid,
        "tid": 0,
        "args": {"name": name},
    }


def thread_name_event(pid: int, tid: int, name: str) -> dict[str, Any]:
    return {
        "name": "thread_name",
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": {"name": name},
    }


def complete_event(
    *,
    name: str,
    start_s: float,
    duration_s: float,
    pid: int,
    tid: int,
    category: str = "request_profile",
    args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": us(start_s),
        "dur": us(max(0.0, float(duration_s))),
        "pid": pid,
        "tid": tid,
        "args": dict(args or {}),
    }


def instant_event(
    *,
    name: str,
    time_s: float,
    pid: int,
    tid: int,
    category: str = "request_profile",
    args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "i",
        "s": "t",
        "ts": us(time_s),
        "pid": pid,
        "tid": tid,
        "args": dict(args or {}),
    }


def flow_event(
    *,
    name: str,
    phase: str,
    time_s: float,
    pid: int,
    tid: int,
    flow_id: int,
    category: str = "flow",
    args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Chrome Trace flow: ``ph='s'`` start, ``ph='f'`` end (with ``bp='e'``)."""
    event: dict[str, Any] = {
        "name": name,
        "cat": category,
        "ph": phase,
        "ts": us(time_s),
        "pid": pid,
        "tid": tid,
        "id": int(flow_id),
        "args": dict(args or {}),
    }
    if phase == "f":
        # Bind to the enclosing slice end so Perfetto/chrome draw the arrow cleanly.
        event["bp"] = "e"
    return event
