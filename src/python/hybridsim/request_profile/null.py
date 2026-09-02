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
        phase: Optional[str] = None,
        scheduled_tokens: Optional[int] = None,
        prefix_hit_tokens: Optional[int] = None,
        n_kernels: Optional[int] = None,
        critical_path_s: Optional[float] = None,
        request_ids: Optional[list[int]] = None,
    ) -> None:
        _ = (
            start_s,
            duration_s,
            replica_id,
            request_id,
            workload_id,
            batch_id,
            request,
            phase,
            scheduled_tokens,
            prefix_hit_tokens,
            n_kernels,
            critical_path_s,
            request_ids,
        )

    def emit_kv_transfer(
        self,
        *,
        start_s: float,
        duration_s: float,
        replica_id: int,
        request_id: int,
        direction: str,
        num_tokens: int = 0,
        block_ids: Optional[list[int]] = None,
    ) -> None:
        _ = (start_s, duration_s, replica_id, request_id, direction, num_tokens, block_ids)

    def emit_engine_kernels(
        self,
        *,
        replica_id: int,
        slices: list[dict[str, Any]],
        flows: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        _ = (replica_id, slices, flows)

    def emit_handoff(
        self,
        *,
        time_s: float,
        request_id: int,
        from_replica_id: int,
        to_replica_id: int,
        request: Any = None,
    ) -> None:
        _ = (time_s, request_id, from_replica_id, to_replica_id, request)

    def emit_request_finish(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        request: Any = None,
    ) -> None:
        _ = (time_s, request_id, replica_id, request)

    def emit_profile_meta(self, meta: dict[str, Any]) -> None:
        _ = meta

    def set_replica_process_name(self, replica_id: int, name: str) -> None:
        _ = (replica_id, name)

    def emit_complete(self, **kwargs: Any) -> None:
        _ = kwargs

    def emit_instant(self, **kwargs: Any) -> None:
        _ = kwargs
