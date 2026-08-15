"""Offline hybridsim driver: schedule ledger + Mooncake Store pool profile."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from hybridsim_infer.schedulers import SchedulerFactory
from hybridsim_infer.kv_system import MooncakeKvStore, VllmKvCacheManager, block_keys_from_tokens
from hybridsim_infer.request import InferenceRequest, RequestStatus

from schedule_alignment.case_loader import CaseSpec
from schedule_alignment.schema import ScheduleStepRecord, normalize_req_id

from .pool_recorder import events as pool_events
from .pool_recorder import record as pool_record
from .pool_recorder import reset as pool_reset
from .pool_recorder import set_step as pool_set_step
from .schema import MooncakePoolEvent, write_pool_profile


def _pool_profile(op: str, **kwargs: Any) -> None:
    """Adapt ``MooncakeKvStore._emit`` kwargs to ``pool_recorder.record``."""
    pool_record(
        op,
        hashes=kwargs.get("hashes") or kwargs.get("keys") or [],
        keys=kwargs.get("keys") or [],
        hit_mask=kwargs.get("hit_mask") or [],
        num_tokens=int(kwargs.get("num_tokens") or 0),
        req_id=str(kwargs.get("req_id") or ""),
        step=kwargs.get("step"),
        block_ids=kwargs.get("block_ids") or [],
    )


def _sched_cfg(case: CaseSpec) -> dict[str, Any]:
    s = case.scheduler
    max_running = int(s.get("max_num_running_reqs", 32))
    max_tokens = int(s.get("max_num_scheduled_tokens", 64))
    if max_tokens < max_running:
        max_tokens = max_running
    return {
        "framework": str(getattr(case, "framework", None) or s.get("framework", "vllm")),
        "max_num_scheduled_tokens": max_tokens,
        "max_num_running_reqs": max_running,
        "tokens_per_step": int(s.get("tokens_per_step", 8)),
        "decode_tokens_per_step": int(s.get("decode_tokens_per_step", 1)),
        "long_prefill_token_threshold": int(s.get("long_prefill_token_threshold", 0)),
        "num_gpu_blocks": int(s.get("num_gpu_blocks", 1024)),
        "block_size": int(s.get("block_size", 16)),
        "reserve_full_isl": bool(s.get("reserve_full_isl", True)),
        "enable_prefix_caching": bool(s.get("enable_prefix_caching", False)),
        "enable_remote_store": bool(s.get("enable_remote_store", True)),
        # Offline Mooncake pool capacity (blocks). ``<=0`` → unlimited DRAM.
        "store_num_blocks": int(s.get("store_num_blocks", 4096)),
    }


def _make_request(spec) -> InferenceRequest:
    rid = (
        int(spec.request_id)
        if str(spec.request_id).isdigit()
        else abs(hash(spec.request_id)) % 10_000_000
    )
    return InferenceRequest(
        request_id=rid,
        arrived_at=float(spec.arrive_step),
        num_prefill_tokens=spec.num_prefill_tokens,
        num_decode_tokens=spec.num_decode_tokens,
        prompt_token_ids=list(spec.prompt_token_ids),
        status=RequestStatus.WAITING,
    )


async def _run_async(
    case: CaseSpec,
) -> tuple[list[ScheduleStepRecord], list[MooncakePoolEvent]]:
    cfg = _sched_cfg(case)
    pool_reset()
    framework = SchedulerFactory.create(
        cfg["framework"],
        tokens_per_step=cfg["tokens_per_step"],
        decode_tokens_per_step=cfg["decode_tokens_per_step"],
        long_prefill_token_threshold=cfg["long_prefill_token_threshold"],
        reserve_full_isl=cfg["reserve_full_isl"],
        enable_prefix_caching=cfg["enable_prefix_caching"],
    )
    kv = VllmKvCacheManager(
        num_gpu_blocks=cfg["num_gpu_blocks"],
        block_size=cfg["block_size"],
        enable_prefix_caching=cfg["enable_prefix_caching"],
    )
    for prefix in case.seed_prefix_cache:
        kv.cache_prefix(list(prefix))

    step_box = {"step": 0}
    # Same engine as DES ``KvStoreActor.store`` — not a test-only mirror.
    pool = MooncakeKvStore(
        num_blocks=cfg["store_num_blocks"],
        block_size=cfg["block_size"],
        profile_fn=_pool_profile,
        profile_step_fn=lambda: step_box["step"],
    )
    # Per-request Mooncake save offset (full blocks already put).
    num_saved_tokens: dict[int, int] = {}

    def remote_lookup(request: InferenceRequest) -> dict[str, Any]:
        if not cfg["enable_remote_store"]:
            return {"hit": False, "num_tokens": 0}
        keys = block_keys_from_tokens(request.prompt_token_ids, cfg["block_size"])
        result = pool.lookup_keys(keys, req_id=str(request.request_id))
        return {
            "hit": bool(result.get("hit")),
            "num_tokens": int(result.get("num_tokens", 0)),
        }

    pending = sorted(case.requests, key=lambda r: (r.arrive_step, r.request_id))
    id_map: dict[int, str] = {}
    waiting: list[InferenceRequest] = []
    running: list[InferenceRequest] = []
    records: list[ScheduleStepRecord] = []
    batch_id = 1
    pending_idx = 0
    idle_streak = 0

    for step in range(case.max_steps):
        step_box["step"] = step
        pool_set_step(step)
        while pending_idx < len(pending) and pending[pending_idx].arrive_step <= step:
            spec = pending[pending_idx]
            pending_idx += 1
            req = _make_request(spec)
            id_map[req.request_id] = normalize_req_id(spec.request_id)
            waiting.append(req)

        if not waiting and not running and pending_idx >= len(pending):
            break

        result = await framework.schedule_step(
            waiting,
            running,
            kv_cache_manager=kv,
            batch_id=batch_id,
            token_budget=cfg["max_num_scheduled_tokens"],
            max_num_running_reqs=cfg["max_num_running_reqs"],
            remote_lookup=remote_lookup if cfg["enable_remote_store"] else None,
        )
        waiting = result.waiting
        running = result.running
        batch_id += 1

        scheduled: dict[str, int] = {}
        new_ids: list[str] = []
        if result.batch is not None:
            for rid, n in result.batch.tokens_per_request.items():
                scheduled[id_map.get(int(rid), str(rid))] = int(n)
            for req in result.batch.requests:
                if req.num_computed_tokens == 0:
                    new_ids.append(id_map.get(req.request_id, str(req.request_id)))

        preempted_ids = [
            id_map.get(p.request_id, str(p.request_id)) for p in result.preempted
        ]
        finished_ids = [
            id_map.get(r.request_id, str(r.request_id)) for r in result.finished_cached
        ]

        if result.batch is not None:
            done_now = framework.on_batch_complete(
                list(result.batch.requests),
                result.batch.tokens_per_request,
                kv,
            )
            bs = cfg["block_size"]
            for req in result.batch.requests:
                # Mooncake write-pool gate: only newly completed full blocks.
                computed = int(req.num_computed_tokens or 0)
                prefill_end = int(req.num_prefill_tokens or 0)
                save_upto = min(computed, prefill_end) if prefill_end > 0 else computed
                aligned = (save_upto // bs) * bs if bs > 0 else 0
                saved = int(num_saved_tokens.get(req.request_id, 0))
                # cdiv(saved + 1, bs) * bs
                chunk_boundary = ((saved + bs) // bs) * bs if bs > 0 else 0
                if aligned >= chunk_boundary and aligned > saved:
                    all_keys = block_keys_from_tokens(
                        req.prompt_token_ids[:aligned], bs
                    )
                    keys = all_keys[saved // bs : aligned // bs]
                    if keys:
                        pool.insert_keys(keys, req_id=str(req.request_id))
                    num_saved_tokens[req.request_id] = aligned
            if done_now:
                done_ids = {r.request_id for r in done_now}
                running = [r for r in running if r.request_id not in done_ids]
                finished_ids.extend(
                    id_map.get(r.request_id, str(r.request_id)) for r in done_now
                )

        for pull in result.remote_pulls:
            req = pull.request
            if req.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
                req.num_computed_tokens = max(
                    req.num_computed_tokens, req.pending_remote_tokens
                )
                req.pending_remote_tokens = 0
                req.status = RequestStatus.WAITING
                keys = block_keys_from_tokens(
                    req.prompt_token_ids[: req.num_computed_tokens],
                    cfg["block_size"],
                )
                pool.get_keys(
                    keys,
                    req_id=str(req.request_id),
                    num_tokens=pull.num_tokens,
                )

        records.append(
            ScheduleStepRecord(
                step=step,
                scheduled_tokens=scheduled,
                new_req_ids=sorted(set(new_ids)),
                preempted_ids=sorted(set(preempted_ids)),
                finished_ids=sorted(set(finished_ids)),
                waiting_ids=[
                    id_map.get(r.request_id, str(r.request_id)) for r in waiting
                ],
                running_ids=[
                    id_map.get(r.request_id, str(r.request_id)) for r in running
                ],
            )
        )

        progressed = bool(
            scheduled or preempted_ids or finished_ids or result.finished_cached
            or result.remote_pulls
        )
        arrived_now = any(p.arrive_step == step for p in case.requests)
        if progressed or arrived_now:
            idle_streak = 0
        else:
            idle_streak += 1
            if idle_streak >= 3 and pending_idx >= len(pending):
                break

        if not waiting and not running and pending_idx >= len(pending):
            break

    return records, pool_events()


def run_hybridsim_store_case(
    case: CaseSpec,
    *,
    schedule_out: Optional[Path] = None,
    pool_out: Optional[Path] = None,
) -> tuple[list[ScheduleStepRecord], list[MooncakePoolEvent]]:
    records, events = asyncio.run(_run_async(case))
    if schedule_out is not None:
        from schedule_alignment.schema import write_ledger

        write_ledger(Path(schedule_out), records)
    if pool_out is not None:
        write_pool_profile(Path(pool_out), events)
    return records, events
