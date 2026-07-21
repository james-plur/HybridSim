"""Record scheduler / batch events as Chrome Trace Format JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _us(seconds: float) -> float:
    return float(seconds) * 1e6


@dataclass
class ScheduleTraceRecorder:
    """Collect scheduling events for chrome://tracing."""

    source: str = "hybridsim"
    run_dir: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _events: list[dict[str, Any]] = field(default_factory=list)
    _request_schedule: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    _next_pid: int = 1
    _cluster_pid: dict[str, int] = field(default_factory=dict)

    def _pid_for_cluster(self, cluster_type: str, replica_id: int = 0) -> int:
        key = f"{cluster_type}/replica_{replica_id}"
        if key not in self._cluster_pid:
            self._cluster_pid[key] = self._next_pid
            self._next_pid += 1
            self._events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": self._cluster_pid[key],
                    "tid": 0,
                    "args": {"name": key},
                }
            )
            self._events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self._cluster_pid[key],
                    "tid": 0,
                    "args": {"name": "batch_execution"},
                }
            )
            self._events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self._cluster_pid[key],
                    "tid": 1,
                    "args": {"name": "scheduler"},
                }
            )
        return self._cluster_pid[key]

    def record_instant(
        self,
        *,
        name: str,
        time_s: float,
        category: str = "scheduler",
        cluster_type: str = "MONOLITHIC",
        replica_id: int = 0,
        request_id: Optional[int] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        pid = self._pid_for_cluster(cluster_type, replica_id)
        payload = dict(args or {})
        if request_id is not None:
            payload["request_id"] = request_id
        event = {
            "name": name,
            "cat": category,
            "ph": "i",
            "s": "t",
            "ts": _us(time_s),
            "pid": pid,
            "tid": 1,
            "args": payload,
        }
        self._events.append(event)
        if request_id is not None:
            self._request_schedule.setdefault(request_id, []).append(
                {
                    "event": name,
                    "time_s": time_s,
                    "cluster_type": cluster_type,
                    "replica_id": replica_id,
                    "args": payload,
                }
            )

    def record_duration(
        self,
        *,
        name: str,
        start_s: float,
        duration_s: float,
        category: str = "batch_execution",
        cluster_type: str = "MONOLITHIC",
        replica_id: int = 0,
        request_ids: Optional[list[int]] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        pid = self._pid_for_cluster(cluster_type, replica_id)
        payload = dict(args or {})
        if request_ids:
            payload["request_ids"] = request_ids
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": _us(start_s),
            "dur": _us(duration_s),
            "pid": pid,
            "tid": 0,
            "args": payload,
        }
        self._events.append(event)
        for request_id in request_ids or []:
            self._request_schedule.setdefault(request_id, []).append(
                {
                    "event": name,
                    "time_s": start_s,
                    "duration_s": duration_s,
                    "cluster_type": cluster_type,
                    "replica_id": replica_id,
                    "args": payload,
                }
            )

    def to_profile(self) -> dict[str, Any]:
        return {
            "traceEvents": list(self._events),
            "displayTimeUnit": "ms",
            "metadata": {
                "source": self.source,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "request_schedule": self._request_schedule,
                **self.metadata,
            },
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_profile(), indent=2), encoding="utf-8")
