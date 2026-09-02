"""vLLM-aligned schedule phases and batch-completion semantics."""

from __future__ import annotations

import inspect
import pprint
import sys
from typing import Any, Optional

from hybridsim_infer.schedulers.factory import InferenceScheduler, RemoteLookupFn
from hybridsim_infer.request import InferenceRequest, RequestStatus
from hybridsim_infer.schedule_types import (
    DecodeChunk,
    PrefillChunk,
    RemoteKvPull,
    ScheduleBatch,
    ScheduleResult,
)


class VllmScheduler(InferenceScheduler):
    """vLLM-like Phase1 RUNNING + Phase2 WAITING scheduler."""

    name = "vllm"

    def __init__(
        self,
        *,
        tokens_per_step: int = 8,
        decode_tokens_per_step: int = 1,
        long_prefill_token_threshold: int = 0,
        reserve_full_isl: bool = True,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.tokens_per_step = int(tokens_per_step)
        self.decode_tokens_per_step = int(decode_tokens_per_step)
        self.long_prefill_token_threshold = int(long_prefill_token_threshold)
        self.reserve_full_isl = bool(reserve_full_isl)
        self.enable_prefix_caching = bool(enable_prefix_caching)

    def chunk_limit(self, req: InferenceRequest) -> int:
        """Tokens to schedule for ``req`` this step (before budget cap)."""
        if req.is_prefill_chunk:
            remaining_prefill = max(0, req.num_prefill_tokens - req.num_computed_tokens)
            if remaining_prefill <= 0:
                return 0
            threshold = (
                self.long_prefill_token_threshold
                if self.long_prefill_token_threshold > 0
                else self.tokens_per_step
            )
            if threshold <= 0:
                return remaining_prefill
            return min(remaining_prefill, threshold)
        remaining = req.remaining_tokens
        if remaining <= 0:
            return 0
        return min(remaining, max(1, self.decode_tokens_per_step))

    @staticmethod
    def _raise_apc_attach_failed(
        request: InferenceRequest,
        local_hit: int,
        kv_cache_manager: Any,
    ) -> None:
        """match() hit but attach_cached_prefix returned None: invariant bug."""
        allocated = getattr(kv_cache_manager, "allocated", {}) or {}
        dump = {
            "request_id": getattr(request, "request_id", None),
            "local_hit": int(local_hit),
            "num_computed_tokens": getattr(request, "num_computed_tokens", None),
            "num_prefill_tokens": getattr(request, "num_prefill_tokens", None),
            "prompt_len": len(getattr(request, "prompt_token_ids", None) or []),
            "free_blocks": getattr(kv_cache_manager, "free_blocks", None),
            "num_gpu_blocks": getattr(kv_cache_manager, "num_gpu_blocks", None),
            "allocated_blocks": {k: len(v) for k, v in allocated.items()},
            "hash_table_size": len(getattr(kv_cache_manager, "_hash_to_block", {}) or {}),
        }
        print("APC attach_cached_prefix failed after match():", file=sys.stderr)
        pprint.pprint(dump, stream=sys.stderr)
        raise RuntimeError(
            "APC attach_cached_prefix failed after match(); "
            f"internal invariant violation: {dump!r}"
        )

    @staticmethod
    def _preempt_fcfs(
        running: list[InferenceRequest],
        *,
        kv_cache_manager: Any,
        scheduled_tokens: dict[int, int],
        scheduled_reqs: list[InferenceRequest],
        req_to_blocks: dict[int, list[Any]],
        token_budget: int,
    ) -> tuple[list[InferenceRequest], int, InferenceRequest]:
        """Preempt newest running request (list tail), matching vLLM FCFS."""
        preempted = running.pop()
        if preempted in scheduled_reqs:
            scheduled_reqs.remove(preempted)
            token_budget += scheduled_tokens.pop(preempted.request_id, 0)
            req_to_blocks.pop(preempted.request_id, None)
        kv_cache_manager.preempt(preempted)
        preempted.status = RequestStatus.PREEMPTED
        return running, token_budget, preempted

    def process_running_queue(
        self,
        running: list[InferenceRequest],
        *,
        kv_cache_manager: Any,
        token_budget: int,
    ) -> tuple[
        list[InferenceRequest],
        list[InferenceRequest],
        dict[int, int],
        dict[int, list[Any]],
        list[InferenceRequest],
        int,
    ]:
        """Phase 1: schedule RUNNING requests."""
        scheduled_reqs: list[InferenceRequest] = []
        num_scheduled_tokens: dict[int, int] = {}
        req_to_blocks: dict[int, list[Any]] = {}
        preempted_reqs: list[InferenceRequest] = []
        req_index = 0

        while req_index < len(running) and token_budget > 0:
            request = running[req_index]
            if request.is_finished() or request.status != RequestStatus.RUNNING:
                req_index += 1
                continue

            num_new = min(self.chunk_limit(request), token_budget)
            if num_new <= 0:
                req_index += 1
                continue

            new_blocks = None
            while True:
                new_blocks = kv_cache_manager.allocate(request, num_new)
                if new_blocks is not None:
                    break
                # Match vLLM: always preempt (incl. self-preempt).
                if not running:
                    break
                running, token_budget, preempted = self._preempt_fcfs(
                    running,
                    kv_cache_manager=kv_cache_manager,
                    scheduled_tokens=num_scheduled_tokens,
                    scheduled_reqs=scheduled_reqs,
                    req_to_blocks=req_to_blocks,
                    token_budget=token_budget,
                )
                preempted_reqs.append(preempted)
                if preempted is request:
                    new_blocks = None
                    break
                if request not in running:
                    new_blocks = None
                    break
                req_index = running.index(request)

            if new_blocks is None:
                break

            scheduled_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new
            req_to_blocks[request.request_id] = new_blocks
            token_budget -= num_new
            req_index += 1

        return (
            running,
            scheduled_reqs,
            num_scheduled_tokens,
            req_to_blocks,
            preempted_reqs,
            token_budget,
        )

    async def process_wait_queue(
        self,
        waiting: list[InferenceRequest],
        running: list[InferenceRequest],
        *,
        kv_cache_manager: Any,
        token_budget: int,
        max_num_running_reqs: int,
        remote_lookup: Optional[RemoteLookupFn] = None,
    ) -> tuple[
        list[InferenceRequest],
        list[InferenceRequest],
        list[InferenceRequest],
        dict[int, int],
        dict[int, list[Any]],
        list[RemoteKvPull],
        list[InferenceRequest],
        int,
        dict[int, int],
    ]:
        """Phase 2: admit WAITING requests."""
        still_waiting: list[InferenceRequest] = []
        newly_scheduled: list[InferenceRequest] = []
        finished_cached: list[InferenceRequest] = []
        num_scheduled_tokens: dict[int, int] = {}
        req_to_blocks: dict[int, list[Any]] = {}
        remote_pulls: list[RemoteKvPull] = []
        prefix_hits: dict[int, int] = {}
        queue = list(waiting)

        for i, request in enumerate(queue):
            if token_budget <= 0 or len(running) >= max_num_running_reqs:
                still_waiting.extend(queue[i:])
                break

            if request.status == RequestStatus.WAIT_FOR_REMOTE_KVS:
                still_waiting.append(request)
                continue

            if request.status == RequestStatus.PREEMPTED:
                request.status = RequestStatus.WAITING

            if request.status != RequestStatus.WAITING:
                still_waiting.append(request)
                continue

            if self.enable_prefix_caching:
                local_hit = kv_cache_manager.match(request)
                if local_hit > request.num_computed_tokens:
                    blocks = kv_cache_manager.attach_cached_prefix(request, local_hit)
                    if blocks is None:
                        self._raise_apc_attach_failed(
                            request, local_hit, kv_cache_manager
                        )
                    request.num_computed_tokens = local_hit
                    prefix_hits[request.request_id] = int(local_hit)
                    request.record_prefix_hit(local_hit)

            if (
                remote_lookup is not None
                and request.num_computed_tokens < request.num_prefill_tokens
            ):
                lookup = remote_lookup(request)
                if inspect.isawaitable(lookup):
                    lookup = await lookup
                if lookup.get("pending"):
                    # Async lookup in flight (≈ vLLM ext_tokens is None): skip, continue.
                    still_waiting.append(request)
                    continue
                hit_n = int(lookup.get("num_tokens", 0)) if lookup.get("hit") else 0
                # Prefill APC ∪ Store only. PD Decode control-plane pull is
                # KV transfer, not a prefix-cache hit.
                pd_kv_transfer = str(lookup.get("mode") or "") == "control_plane" or bool(
                    (request.kv_transfer_params or {}).get("do_remote_prefill")
                )
                if not pd_kv_transfer:
                    request.record_prefix_hit(
                        max(int(request.num_computed_tokens), hit_n)
                    )
                if hit_n > request.num_computed_tokens:
                    gain = hit_n - request.num_computed_tokens
                    blocks = kv_cache_manager.allocate(request, gain)
                    if blocks is None:
                        still_waiting.append(request)
                        still_waiting.extend(queue[i + 1 :])
                        break
                    token_ids = list(request.prompt_token_ids[:hit_n])
                    request.status = RequestStatus.WAIT_FOR_REMOTE_KVS
                    request.pending_remote_tokens = hit_n
                    block_ids = [int(b.block_id) for b in (blocks or [])]
                    remote_pulls.append(
                        RemoteKvPull(
                            request=request,
                            num_tokens=gain,
                            token_ids=token_ids,
                            block_ids=block_ids,
                        )
                    )
                    still_waiting.append(request)
                    continue

            if request.is_finished():
                request.status = RequestStatus.FINISHED
                request.completed = True
                if self.enable_prefix_caching:
                    kv_cache_manager.cache_request_prefix(request)
                kv_cache_manager.free(request)
                finished_cached.append(request)
                continue

            num_new = min(self.chunk_limit(request), token_budget)
            if num_new <= 0:
                still_waiting.append(request)
                continue

            if self.reserve_full_isl:
                full_tokens = int(
                    getattr(request, "num_tokens", request.num_prefill_tokens)
                )
                if not kv_cache_manager.can_fit(request, full_tokens):
                    still_waiting.append(request)
                    still_waiting.extend(queue[i + 1 :])
                    break

            new_blocks = kv_cache_manager.allocate(request, num_new)
            if new_blocks is None:
                still_waiting.append(request)
                still_waiting.extend(queue[i + 1 :])
                break

            request.status = RequestStatus.RUNNING
            running.append(request)
            newly_scheduled.append(request)
            num_scheduled_tokens[request.request_id] = num_new
            req_to_blocks[request.request_id] = new_blocks
            token_budget -= num_new

        return (
            still_waiting,
            running,
            newly_scheduled,
            num_scheduled_tokens,
            req_to_blocks,
            remote_pulls,
            finished_cached,
            token_budget,
            prefix_hits,
        )

    @staticmethod
    def build_batch(
        scheduled_tokens: dict[int, int],
        requests_by_id: dict[int, InferenceRequest],
        req_to_blocks: dict[int, list[Any]],
        *,
        batch_id: int,
    ) -> Optional[ScheduleBatch]:
        if not scheduled_tokens:
            return None
        chunks: list[Any] = []
        requests: list[InferenceRequest] = []
        for rid, n in scheduled_tokens.items():
            req = requests_by_id[rid]
            requests.append(req)
            if req.is_prefill_chunk or req.num_computed_tokens < req.num_prefill_tokens:
                remaining_prefill = max(
                    0, req.num_prefill_tokens - req.num_computed_tokens
                )
                if remaining_prefill > 0:
                    chunks.append(
                        PrefillChunk(
                            request=req, num_tokens=min(n, remaining_prefill)
                        )
                    )
                    decode_n = n - min(n, remaining_prefill)
                    if decode_n > 0:
                        chunks.append(DecodeChunk(request=req, num_tokens=decode_n))
                else:
                    chunks.append(DecodeChunk(request=req, num_tokens=n))
            else:
                chunks.append(DecodeChunk(request=req, num_tokens=n))
        return ScheduleBatch(
            batch_id=batch_id,
            chunks=chunks,
            requests=requests,
            tokens_per_request=dict(scheduled_tokens),
            req_to_new_blocks=dict(req_to_blocks),
        )

    async def schedule_step(
        self,
        waiting: list[InferenceRequest],
        running: list[InferenceRequest],
        *,
        kv_cache_manager: Any,
        batch_id: int,
        token_budget: int,
        max_num_running_reqs: int,
        remote_lookup: Optional[RemoteLookupFn] = None,
    ) -> ScheduleResult:
        budget = token_budget if token_budget > 0 else 10**9

        (
            running,
            scheduled_running,
            tokens_r,
            blocks_r,
            preempted,
            budget,
        ) = self.process_running_queue(
            list(running),
            kv_cache_manager=kv_cache_manager,
            token_budget=budget,
        )

        waiting = list(waiting)
        for p in reversed(preempted):
            p.status = RequestStatus.WAITING
            waiting.insert(0, p)

        tokens_w: dict[int, int] = {}
        blocks_w: dict[int, list[Any]] = {}
        remote_pulls: list[RemoteKvPull] = []
        newly: list[InferenceRequest] = []
        finished_cached: list[InferenceRequest] = []
        prefix_hits: dict[int, int] = {}

        if not preempted:
            (
                waiting,
                running,
                newly,
                tokens_w,
                blocks_w,
                remote_pulls,
                finished_cached,
                budget,
                prefix_hits,
            ) = await self.process_wait_queue(
                waiting,
                running,
                kv_cache_manager=kv_cache_manager,
                token_budget=budget,
                max_num_running_reqs=max_num_running_reqs,
                remote_lookup=remote_lookup,
            )

        merged_tokens = {**tokens_r, **tokens_w}
        merged_blocks = {**blocks_r, **blocks_w}
        by_id = {r.request_id: r for r in scheduled_running + newly}
        sched_batch = self.build_batch(
            merged_tokens, by_id, merged_blocks, batch_id=batch_id
        )

        return ScheduleResult(
            waiting=waiting,
            running=running,
            batch=sched_batch,
            remote_pulls=remote_pulls,
            preempted=preempted,
            finished_cached=finished_cached,
            prefix_hits=prefix_hits,
        )

    def on_batch_complete(
        self,
        requests: list[InferenceRequest],
        tokens_per_request: dict[int, int],
        kv_cache_manager: Any,
    ) -> list[InferenceRequest]:
        """Advance after forward; last prefill step also samples +1 output token."""
        finished: list[InferenceRequest] = []
        for req in requests:
            n = int(tokens_per_request.get(req.request_id, 0))
            was_prefill = req.num_computed_tokens < req.num_prefill_tokens
            req.num_computed_tokens += n
            if (
                was_prefill
                and req.num_computed_tokens >= req.num_prefill_tokens
                and req.num_decode_tokens > 0
                and req.num_output_tokens < req.num_decode_tokens
            ):
                req.num_output_tokens += 1
                if req.num_computed_tokens < req.num_tokens_with_output:
                    req.num_computed_tokens += 1
            elif (
                not was_prefill
                and n > 0
                and req.num_output_tokens < req.num_decode_tokens
            ):
                req.num_output_tokens += n
            if req.is_finished():
                req.status = RequestStatus.FINISHED
                req.completed = True
                if self.enable_prefix_caching:
                    kv_cache_manager.cache_request_prefix(req)
                kv_cache_manager.free(req)
                finished.append(req)
        return finished
