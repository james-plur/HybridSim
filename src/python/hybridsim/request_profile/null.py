"""No-op request profile session (profiling disabled)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class NullRequestProfileSession:
    """Drop-in when ``enable_request_profile`` is False."""

    enabled: bool = False
    output_path: Optional[Path] = None
    dropped: int = 0

    def start(self) -> None:
        return

    def stop(self, timeout: float = 5.0) -> Optional[Path]:
        _ = timeout
        return None

    def emit_cluster_schedule(self, *, time_s: float) -> None:
        _ = time_s

    def emit_dispatch(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        kind: str = "arrive",
        request: Any = None,
    ) -> None:
        _ = (time_s, request_id, replica_id, kind, request)

    def emit_replica_enqueue(
        self,
        *,
        time_s: float,
        replica_id: int,
        request_id: int,
        request: Any = None,
    ) -> None:
        _ = (time_s, replica_id, request_id, request)

    def emit_request_meta(
        self,
        *,
        request: Any = None,
        meta: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        _ = (request, meta, extra)

    def emit_replica_schedule(
        self,
        *,
        time_s: float,
        replica_id: int,
        batch_id: Optional[int] = None,
        request_ids: Optional[list[int]] = None,
    ) -> None:
        _ = (time_s, replica_id, batch_id, request_ids)

    def emit_engine_req(
        self,
        *,
        start_s: float,
        duration_s: float,
        replica_id: int,
        request_id: int,
        workload_id: int,
        batch_id: int,
        request: Any = None,
    ) -> None:
        _ = (start_s, duration_s, replica_id, request_id, workload_id, batch_id, request)

    def emit_kv_transfer(
        self,
        *,
        start_s: float,
        duration_s: float,
        replica_id: int,
        request_id: int,
        direction: str,
        num_tokens: int = 0,
    ) -> None:
        _ = (start_s, duration_s, replica_id, request_id, direction, num_tokens)

    def emit_complete(self, **kwargs: Any) -> None:
        _ = kwargs

    def emit_instant(self, **kwargs: Any) -> None:
        _ = kwargs
