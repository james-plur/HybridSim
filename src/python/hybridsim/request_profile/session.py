"""Request profile session: parent emits events; child writes Chrome Trace JSON."""

from __future__ import annotations

import atexit
import multiprocessing as mp
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Optional, Protocol, Union

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
    replica_pid,
)
from hybridsim.request_profile.null import NullRequestProfileSession
from hybridsim.request_profile.request_meta import snapshot_request_meta
from hybridsim.request_profile.writer import writer_main


class RequestProfileLike(Protocol):
    enabled: bool
    output_path: Optional[Path]
    dropped: int

    def start(self) -> None: ...
    def stop(self, timeout: float = 5.0) -> Optional[Path]: ...
    def emit_cluster_schedule(self, *, time_s: float) -> None: ...
    def emit_dispatch(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        kind: str = "arrive",
        request: Any = None,
    ) -> None: ...
    def emit_replica_enqueue(
        self,
        *,
        time_s: float,
        replica_id: int,
        request_id: int,
        request: Any = None,
    ) -> None: ...
    def emit_request_meta(
        self,
        *,
        request: Any = None,
        meta: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None: ...
    def emit_replica_schedule(
        self,
        *,
        time_s: float,
        replica_id: int,
        batch_id: Optional[int] = None,
        request_ids: Optional[list[int]] = None,
    ) -> None: ...
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
    ) -> None: ...
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
    ) -> None: ...
    def emit_engine_kernels(
        self,
        *,
        replica_id: int,
        slices: list[dict[str, Any]],
        flows: Optional[list[dict[str, Any]]] = None,
    ) -> None: ...
    def emit_handoff(
        self,
        *,
        time_s: float,
        request_id: int,
        from_replica_id: int,
        to_replica_id: int,
        request: Any = None,
    ) -> None: ...
    def emit_request_finish(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        request: Any = None,
    ) -> None: ...
    def emit_profile_meta(self, meta: dict[str, Any]) -> None: ...
    def set_replica_process_name(self, replica_id: int, name: str) -> None: ...


RequestProfileSessionT = Union["RequestProfileSession", NullRequestProfileSession]


def default_profile_dir() -> Path:
    """``<hybridsim repo root>/profile`` (package lives at ``src/python/hybridsim``)."""
    # hybridsim/request_profile/session.py → parents[4] = repo root
    return Path(__file__).resolve().parents[4] / "profile"


def resolve_profile_path(
    *,
    request_profile_path: Optional[Path] = None,
    request_profile_dir: Optional[Path] = None,
) -> Path:
    if request_profile_path is not None:
        return Path(request_profile_path)
    directory = Path(request_profile_dir) if request_profile_dir else default_profile_dir()
    return directory / "request_profile.json"


class RequestProfileSession:
    """Main-process handle: Queue + writer Process."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.enabled = True
        self.dropped = 0
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._stopped = False
        self._started = False
        self._next_flow_id = 1
        #: Cluster Dispatch → ReplicaEnqueue (FIFO per request; arrive then handoff).
        self._pending_dispatch_flows: dict[int, Deque[int]] = defaultdict(deque)
        #: ReplicaSchedule → EngineReq keyed by (replica_id, request_id).
        self._pending_schedule_flows: dict[tuple[int, int], Deque[int]] = defaultdict(
            deque
        )
        self._replica_process_names: dict[int, str] = {}

    def start(self) -> None:
        if self._started:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=100_000)
        self._process = ctx.Process(
            target=writer_main,
            args=(self._queue, str(self.output_path)),
            name="hybridsim-request-profile",
            daemon=False,
        )
        self._process.start()
        self._started = True
        self._stopped = False
        atexit.register(self._atexit_stop)

    def _atexit_stop(self) -> None:
        try:
            self.stop(timeout=2.0)
        except Exception:
            pass

    def _alloc_flow_id(self) -> int:
        fid = self._next_flow_id
        self._next_flow_id += 1
        return fid

    def set_replica_process_name(self, replica_id: int, name: str) -> None:
        self._replica_process_names[int(replica_id)] = str(name)

    def emit_profile_meta(self, meta: dict[str, Any]) -> None:
        self._put({"kind": MSG_PROFILE_META, "meta": dict(meta)})

    def _put(self, msg: dict[str, Any]) -> None:
        if self._queue is None or self._stopped:
            return
        rid = msg.get("replica_id")
        if rid is not None and not msg.get("process_name"):
            name = self._replica_process_names.get(int(rid))
            if name:
                msg["process_name"] = name
        try:
            self._queue.put_nowait(msg)
        except Exception:
            self.dropped += 1

    def emit_complete(
        self,
        *,
        name: str,
        start_s: float,
        duration_s: float,
        pid: int,
        tid: int,
        category: str = "request_profile",
        args: Optional[dict[str, Any]] = None,
        replica_id: Optional[int] = None,
        track: Optional[str] = None,
        thread_name: Optional[str] = None,
        process_name: Optional[str] = None,
    ) -> None:
        msg: dict[str, Any] = {
            "kind": MSG_COMPLETE,
            "name": name,
            "start_s": float(start_s),
            "duration_s": float(duration_s),
            "pid": int(pid),
            "tid": int(tid),
            "category": category,
            "args": dict(args or {}),
            "replica_id": replica_id,
            "track": track,
        }
        if thread_name:
            msg["thread_name"] = str(thread_name)
        if process_name:
            msg["process_name"] = str(process_name)
        self._put(msg)

    def emit_instant(
        self,
        *,
        name: str,
        time_s: float,
        pid: int,
        tid: int,
        category: str = "request_profile",
        args: Optional[dict[str, Any]] = None,
        replica_id: Optional[int] = None,
        track: Optional[str] = None,
        thread_name: Optional[str] = None,
        process_name: Optional[str] = None,
    ) -> None:
        msg: dict[str, Any] = {
            "kind": MSG_INSTANT,
            "name": name,
            "time_s": float(time_s),
            "pid": int(pid),
            "tid": int(tid),
            "category": category,
            "args": dict(args or {}),
            "replica_id": replica_id,
            "track": track,
        }
        if thread_name:
            msg["thread_name"] = str(thread_name)
        if process_name:
            msg["process_name"] = str(process_name)
        self._put(msg)

    def emit_flow(
        self,
        *,
        name: str,
        phase: str,
        time_s: float,
        pid: int,
        tid: int,
        flow_id: int,
        category: str = "flow",
        args: Optional[dict[str, Any]] = None,
        replica_id: Optional[int] = None,
        track: Optional[str] = None,
        thread_name: Optional[str] = None,
        process_name: Optional[str] = None,
    ) -> None:
        msg: dict[str, Any] = {
            "kind": MSG_FLOW,
            "name": name,
            "phase": phase,
            "time_s": float(time_s),
            "pid": int(pid),
            "tid": int(tid),
            "flow_id": int(flow_id),
            "category": category,
            "args": dict(args or {}),
            "replica_id": replica_id,
            "track": track,
        }
        if thread_name:
            msg["thread_name"] = str(thread_name)
        if process_name:
            msg["process_name"] = str(process_name)
        self._put(msg)

    def emit_cluster_schedule(self, *, time_s: float) -> None:
        self.emit_complete(
            name="ClusterSchedule",
            start_s=time_s,
            duration_s=0.0,
            pid=PID_CLUSTER,
            tid=TID_CLUSTER_SCHEDULE,
            args={},
            track="cluster",
        )

    def emit_request_meta(
        self,
        *,
        request: Any = None,
        meta: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if meta is None:
            if request is None:
                return
            meta = snapshot_request_meta(request, extra=extra)
        elif extra:
            meta = dict(meta)
            meta.update(extra)
        self._put({"kind": MSG_REQUEST_META, "meta": dict(meta)})

    def emit_dispatch(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        kind: str = "arrive",
        request: Any = None,
    ) -> None:
        rid = int(request_id)
        req_args: dict[str, Any] = {
            "request_id": rid,
            "replica_id": int(replica_id),
            "kind": kind,
        }
        if request is not None:
            snap = snapshot_request_meta(request, extra={"dispatch_kind": kind})
            self.emit_request_meta(meta=snap)
            req_args.update(
                {
                    "arrived_at": snap["arrived_at"],
                    "num_prefill_tokens": snap["num_prefill_tokens"],
                    "num_decode_tokens": snap["num_decode_tokens"],
                    "prompt_len": snap["prompt_len"],
                }
            )
        self.emit_complete(
            name="Dispatch",
            start_s=time_s,
            duration_s=0.0,
            pid=PID_CLUSTER,
            tid=TID_CLUSTER_DISPATCH,
            args=req_args,
            track="cluster",
        )
        flow_id = self._alloc_flow_id()
        self._pending_dispatch_flows[rid].append(flow_id)
        self.emit_flow(
            name="ClusterToReplica",
            phase="s",
            time_s=time_s,
            pid=PID_CLUSTER,
            tid=TID_CLUSTER_DISPATCH,
            flow_id=flow_id,
            args={"request_id": rid, "replica_id": int(replica_id), "kind": kind},
            track="cluster",
        )

    def emit_replica_enqueue(
        self,
        *,
        time_s: float,
        replica_id: int,
        request_id: int,
        request: Any = None,
    ) -> None:
        rid = int(request_id)
        pid = replica_pid(replica_id)
        args: dict[str, Any] = {"request_id": rid}
        if request is not None:
            snap = snapshot_request_meta(request)
            self.emit_request_meta(meta=snap)
            args.update(
                {
                    "arrived_at": snap["arrived_at"],
                    "num_prefill_tokens": snap["num_prefill_tokens"],
                    "num_decode_tokens": snap["num_decode_tokens"],
                    "prompt_len": snap["prompt_len"],
                }
            )
        self.emit_complete(
            name="ReplicaEnqueue",
            start_s=time_s,
            duration_s=0.0,
            pid=pid,
            tid=TID_REPLICA_SCHEDULE,
            args=args,
            replica_id=int(replica_id),
        )
        pending = self._pending_dispatch_flows.get(rid)
        if pending:
            flow_id = pending.popleft()
            self.emit_flow(
                name="ClusterToReplica",
                phase="f",
                time_s=time_s,
                pid=pid,
                tid=TID_REPLICA_SCHEDULE,
                flow_id=flow_id,
                args={"request_id": rid, "replica_id": int(replica_id)},
                replica_id=int(replica_id),
            )

    def emit_replica_schedule(
        self,
        *,
        time_s: float,
        replica_id: int,
        batch_id: Optional[int] = None,
        request_ids: Optional[list[int]] = None,
    ) -> None:
        args: dict[str, Any] = {}
        if batch_id is not None:
            args["batch_id"] = int(batch_id)
        ids = [int(x) for x in (request_ids or [])]
        if ids:
            args["request_ids"] = ids
        pid = replica_pid(replica_id)
        self.emit_complete(
            name="ReplicaSchedule",
            start_s=time_s,
            duration_s=0.0,
            pid=pid,
            tid=TID_REPLICA_SCHEDULE,
            args=args,
            replica_id=int(replica_id),
        )
        for req_id in ids:
            flow_id = self._alloc_flow_id()
            self._pending_schedule_flows[(int(replica_id), req_id)].append(flow_id)
            self.emit_flow(
                name="ScheduleToEngine",
                phase="s",
                time_s=time_s,
                pid=pid,
                tid=TID_REPLICA_SCHEDULE,
                flow_id=flow_id,
                args={
                    "request_id": req_id,
                    "replica_id": int(replica_id),
                    "batch_id": int(batch_id) if batch_id is not None else None,
                },
                replica_id=int(replica_id),
            )

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
        rid = int(request_id)
        pid = replica_pid(replica_id)
        args: dict[str, Any] = {
            "request_id": rid,
            "workload_id": int(workload_id),
            "batch_id": int(batch_id),
        }
        if phase is not None:
            args["phase"] = str(phase)
        if scheduled_tokens is not None:
            args["scheduled_tokens"] = int(scheduled_tokens)
        if prefix_hit_tokens is not None:
            args["prefix_hit_tokens"] = int(prefix_hit_tokens)
        if n_kernels is not None:
            args["n_kernels"] = int(n_kernels)
        if critical_path_s is not None:
            args["critical_path_s"] = float(critical_path_s)
        if request_ids:
            args["request_ids"] = [int(x) for x in request_ids]
        if request is not None:
            snap = snapshot_request_meta(request)
            args.update(
                {
                    "num_prefill_tokens": snap["num_prefill_tokens"],
                    "num_decode_tokens": snap["num_decode_tokens"],
                    "num_computed_tokens": snap["num_computed_tokens"],
                    "prefix_hit_tokens": snap.get(
                        "prefix_hit_tokens", args.get("prefix_hit_tokens", 0)
                    ),
                }
            )
        self.emit_complete(
            name="EngineReq",
            start_s=start_s,
            duration_s=duration_s,
            pid=pid,
            tid=TID_REPLICA_ENGINE,
            args=args,
            replica_id=int(replica_id),
        )
        pending = self._pending_schedule_flows.get((int(replica_id), rid))
        if pending:
            flow_id = pending.popleft()
            self.emit_flow(
                name="ScheduleToEngine",
                phase="f",
                time_s=start_s,
                pid=pid,
                tid=TID_REPLICA_ENGINE,
                flow_id=flow_id,
                args={
                    "request_id": rid,
                    "replica_id": int(replica_id),
                    "workload_id": int(workload_id),
                    "batch_id": int(batch_id),
                },
                replica_id=int(replica_id),
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
        name = "KvPull" if str(direction) == "pull" else "KvPush"
        args: dict[str, Any] = {
            "request_id": int(request_id),
            "direction": str(direction),
            "num_tokens": int(num_tokens),
        }
        if block_ids:
            args["block_ids"] = [int(x) for x in block_ids]
        self.emit_complete(
            name=name,
            start_s=start_s,
            duration_s=duration_s,
            pid=replica_pid(replica_id),
            tid=TID_REPLICA_ENGINE,
            args=args,
            replica_id=int(replica_id),
        )

    def emit_engine_kernels(
        self,
        *,
        replica_id: int,
        slices: list[dict[str, Any]],
        flows: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        pid = replica_pid(replica_id)
        for sl in slices:
            self.emit_complete(
                name=str(sl["name"]),
                start_s=float(sl["start_s"]),
                duration_s=float(sl.get("duration_s", 0.0)),
                pid=pid,
                tid=int(sl["tid"]),
                category=str(sl.get("category", "op_kernel")),
                args=dict(sl.get("args") or {}),
                replica_id=int(replica_id),
                thread_name=sl.get("thread_name"),
            )
        for fl in flows or []:
            flow_id = self._alloc_flow_id()
            args = dict(fl.get("args") or {})
            self.emit_flow(
                name=str(fl.get("name", "KernelDep")),
                phase="s",
                time_s=float(fl["start_s"]),
                pid=pid,
                tid=int(fl["src_tid"]),
                flow_id=flow_id,
                args=args,
                replica_id=int(replica_id),
            )
            self.emit_flow(
                name=str(fl.get("name", "KernelDep")),
                phase="f",
                time_s=float(fl["end_s"]),
                pid=pid,
                tid=int(fl["dst_tid"]),
                flow_id=flow_id,
                args=args,
                replica_id=int(replica_id),
            )

    def emit_handoff(
        self,
        *,
        time_s: float,
        request_id: int,
        from_replica_id: int,
        to_replica_id: int,
        request: Any = None,
    ) -> None:
        args: dict[str, Any] = {
            "request_id": int(request_id),
            "from_replica_id": int(from_replica_id),
            "to_replica_id": int(to_replica_id),
        }
        if request is not None:
            snap = snapshot_request_meta(request)
            args["prefix_hit_tokens"] = snap.get("prefix_hit_tokens", 0)
        self.emit_instant(
            name="Handoff",
            time_s=time_s,
            pid=PID_CLUSTER,
            tid=TID_CLUSTER_DISPATCH,
            args=args,
            track="cluster",
        )

    def emit_request_finish(
        self,
        *,
        time_s: float,
        request_id: int,
        replica_id: int,
        request: Any = None,
    ) -> None:
        args: dict[str, Any] = {
            "request_id": int(request_id),
            "replica_id": int(replica_id),
        }
        if request is not None:
            snap = snapshot_request_meta(request)
            args.update(
                {
                    "prefix_hit_tokens": snap.get("prefix_hit_tokens", 0),
                    "num_prefill_tokens": snap.get("num_prefill_tokens"),
                    "num_decode_tokens": snap.get("num_decode_tokens"),
                    "finished_at": snap.get("finished_at", time_s),
                }
            )
        self.emit_instant(
            name="RequestFinish",
            time_s=time_s,
            pid=PID_CLUSTER,
            tid=TID_CLUSTER_SCHEDULE,
            args=args,
            track="cluster",
        )

    def stop(self, timeout: float = 5.0) -> Optional[Path]:
        if self._stopped or not self._started:
            self._stopped = True
            return self.output_path if self.output_path.exists() else None
        self._stopped = True
        if self._queue is not None:
            try:
                self._queue.put({"kind": MSG_STOP}, timeout=1.0)
            except Exception:
                try:
                    self._queue.put_nowait({"kind": MSG_STOP})
                except Exception:
                    pass
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:
                pass
        return self.output_path if self.output_path.exists() else None

    @property
    def process_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())


def create_request_profile_session(
    *,
    enabled: bool = False,
    request_profile_path: Optional[Path] = None,
    request_profile_dir: Optional[Path] = None,
) -> RequestProfileSessionT:
    if not enabled:
        return NullRequestProfileSession()
    path = resolve_profile_path(
        request_profile_path=request_profile_path,
        request_profile_dir=request_profile_dir,
    )
    session = RequestProfileSession(path)
    session.start()
    return session
