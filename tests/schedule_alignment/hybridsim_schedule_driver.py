"""Offline hybridsim schedule driver: call framework.schedule_step in a loop."""

from __future__ import annotations

from typing import Any

from hybridsim_infer.frameworks import FrameworkFactory
from hybridsim_infer.kv_cache import KvCacheManager
from hybridsim_infer.request import InferenceRequest, RequestStatus

from .case_loader import CaseSpec
from .schema import ScheduleStepRecord, normalize_req_id


def _sched_cfg(case: CaseSpec) -> dict[str, Any]:
    s = case.scheduler
    max_running = int(s.get("max_num_running_reqs", 32))
    max_tokens = int(s.get("max_num_scheduled_tokens", 64))
    # vLLM requires max_num_batched_tokens >= max_num_seqs.
    if max_tokens < max_running:
        max_tokens = max_running
    return {
        "framework": str(
            getattr(case, "framework", None)
            or s.get("framework", "vllm")
        ),
        "max_num_scheduled_tokens": max_tokens,
        "max_num_running_reqs": max_running,
        "tokens_per_step": int(s.get("tokens_per_step", 8)),
        "decode_tokens_per_step": int(s.get("decode_tokens_per_step", 1)),
        "long_prefill_token_threshold": int(s.get("long_prefill_token_threshold", 0)),
        "num_gpu_blocks": int(s.get("num_gpu_blocks", 1024)),
        "block_size": int(s.get("block_size", 16)),
        "reserve_full_isl": bool(s.get("reserve_full_isl", True)),
        "enable_prefix_caching": bool(s.get("enable_prefix_caching", False)),
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


def run_hybridsim_schedule(case: CaseSpec) -> list[ScheduleStepRecord]:
    cfg = _sched_cfg(case)
    framework = FrameworkFactory.create(
        cfg["framework"],
        tokens_per_step=cfg["tokens_per_step"],
        decode_tokens_per_step=cfg["decode_tokens_per_step"],
        long_prefill_token_threshold=cfg["long_prefill_token_threshold"],
        reserve_full_isl=cfg["reserve_full_isl"],
        enable_prefix_caching=cfg["enable_prefix_caching"],
    )
    kv = KvCacheManager(
        num_gpu_blocks=cfg["num_gpu_blocks"],
        block_size=cfg["block_size"],
    )
    for prefix in case.seed_prefix_cache:
        kv.cache_prefix(list(prefix))

    pending = sorted(case.requests, key=lambda r: (r.arrive_step, r.request_id))
    id_map: dict[int, str] = {}
    waiting: list[InferenceRequest] = []
    running: list[InferenceRequest] = []
    finished: list[InferenceRequest] = []
    records: list[ScheduleStepRecord] = []
    batch_id = 1
    pending_idx = 0
    idle_streak = 0

    for step in range(case.max_steps):
        while pending_idx < len(pending) and pending[pending_idx].arrive_step <= step:
            spec = pending[pending_idx]
            pending_idx += 1
            req = _make_request(spec)
            id_map[req.request_id] = normalize_req_id(spec.request_id)
            waiting.append(req)

        if not waiting and not running and pending_idx >= len(pending):
            break

        result = framework.schedule_step(
            waiting,
            running,
            kv_cache_manager=kv,
            batch_id=batch_id,
            token_budget=cfg["max_num_scheduled_tokens"],
            max_num_running_reqs=cfg["max_num_running_reqs"],
            remote_lookup=None,
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
        for r in result.finished_cached:
            finished.append(r)

        if result.batch is not None:
            done_now = framework.on_batch_complete(
                list(result.batch.requests),
                result.batch.tokens_per_request,
                kv,
            )
            if done_now:
                done_ids = {r.request_id for r in done_now}
                running = [r for r in running if r.request_id not in done_ids]
                finished.extend(done_now)
                finished_ids.extend(
                    id_map.get(r.request_id, str(r.request_id)) for r in done_now
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

    return records


# Back-compat alias.
run_hybridsim_schedule_sync = run_hybridsim_schedule
