"""Offline vLLM schedule driver: keep Scheduler, intercept execution via fake ModelRunnerOutput.

Requires: torch + PYTHONPATH/VLLM_ROOT pointing at a vLLM tree (e.g. /home/y_luchenda/vllm-main).
Does **not** start Engine/GPU kernels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


def ensure_vllm_path() -> Optional[Path]:
    root = Path(os.environ.get("VLLM_ROOT", "/home/y_luchenda/vllm-main"))
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Local editable tree may lack generated _version.py
    ver = root / "vllm" / "_version.py"
    if root.exists() and not ver.exists():
        ver.write_text(
            '__version__ = "0.0.0+local"\n__version_tuple__ = (0, 0, 0)\n',
            encoding="utf-8",
        )
    return root if root.exists() else None


def vllm_available() -> bool:
    ensure_vllm_path()
    try:
        import torch  # noqa: F401
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

        return True
    except Exception:
        return False


def _default_model() -> str:
    # Prefer a local tiny HF config (no network). Override with VLLM_ALIGN_MODEL.
    default = Path(__file__).resolve().parent / "dummy_hf_model"
    return os.environ.get("VLLM_ALIGN_MODEL", str(default))


def run_vllm_schedule(case) -> list:
    """Run vLLM Scheduler offline; raise if deps missing."""
    ensure_vllm_path()
    if not vllm_available():
        raise RuntimeError(
            "vLLM/torch not available. Install torch and set VLLM_ROOT "
            "(default /home/y_luchenda/vllm-main)."
        )

    # Stay offline unless user explicitly allows downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Align NONE_HASH / block hashes with hybridsim block_keys.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    import torch
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    from .case_loader import CaseSpec
    from .schema import ScheduleStepRecord, normalize_req_id

    assert isinstance(case, CaseSpec)
    s = case.scheduler
    max_num_seqs = int(s.get("max_num_running_reqs", s.get("max_num_seqs", 32)))
    max_batched = int(s.get("max_num_scheduled_tokens", 64))
    # vLLM SchedulerConfig: max_num_batched_tokens >= max_num_seqs
    if max_batched < max_num_seqs:
        max_batched = max_num_seqs
    block_size = int(s.get("block_size", 16))
    num_blocks = int(s.get("num_gpu_blocks", 1024))
    long_prefill = int(s.get("long_prefill_token_threshold", 0))
    if long_prefill <= 0:
        long_prefill = int(s.get("tokens_per_step", 0))

    # Force CPU + offline HF for schedule-only intercept path.
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    init_none_hash(sha256)
    model_config = ModelConfig(
        model=_default_model(),
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    max_model_len = max(max_batched, 2048)
    for req in case.requests:
        max_model_len = max(
            max_model_len, req.num_prefill_tokens + req.num_decode_tokens + 16
        )
    # gpt2 default max is 1024; bump via scheduler max_model_len only — ModelConfig
    # may clamp. Ensure we don't exceed for alignment cases with short prompts.
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched,
        max_model_len=min(max_model_len, getattr(model_config, "max_model_len", max_model_len) or max_model_len),
        long_prefill_token_threshold=long_prefill,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
        watermark=0.0,
        scheduler_reserve_full_isl=bool(s.get("reserve_full_isl", True)),
    )
    # If model max_model_len is smaller than our prompts, override carefully.
    try:
        if model_config.max_model_len < max_model_len:
            # Recreate scheduler_config with model limit; cases should stay small.
            scheduler_config = SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_batched,
                max_model_len=model_config.max_model_len,
                long_prefill_token_threshold=long_prefill,
                enable_chunked_prefill=True,
                async_scheduling=False,
                is_encoder_decoder=model_config.is_encoder_decoder,
                watermark=0.0,
                scheduler_reserve_full_isl=bool(s.get("reserve_full_isl", True)),
            )
    except Exception:
        pass

    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=bool(s.get("enable_prefix_caching", False)),
    )
    # Force CPU device — schedule-only path; no CUDA kernels.
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    block_hasher = get_request_block_hasher(block_size, sha256)
    eos = 50256
    pending = sorted(case.requests, key=lambda r: (r.arrive_step, r.request_id))
    pending_idx = 0
    records: list[ScheduleStepRecord] = []
    live: dict[str, Request] = {}
    prev_waiting_preempted: set[str] = set()
    idle_streak = 0

    def _build_request(spec) -> Request:
        sp = SamplingParams(
            ignore_eos=True,
            max_tokens=max(1, spec.num_decode_tokens),
        )
        sp.update_from_generation_config({}, eos)
        prompt = list(spec.prompt_token_ids) or [0] * spec.num_prefill_tokens
        # Clamp prompt to model max if needed
        max_len = int(getattr(model_config, "max_model_len", 1024) or 1024)
        if len(prompt) + spec.num_decode_tokens > max_len:
            keep = max(1, max_len - spec.num_decode_tokens - 1)
            prompt = prompt[:keep]
        return Request(
            request_id=normalize_req_id(spec.request_id),
            prompt_token_ids=prompt,
            sampling_params=sp,
            pooling_params=None,
            mm_features=None,
            block_hasher=block_hasher,
        )

    def _fake_output(scheduler_output) -> ModelRunnerOutput:
        """Intercept execution: emulate worker after schedule() advanced computed.

        Note: vLLM ``schedule()`` already bumps ``num_computed_tokens`` by the
        scheduled amount before the model runs; ``update_from_output`` then
        appends sampled decode tokens.
        """
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        sampled: list[list[int]] = []
        for rid in req_ids:
            req = scheduler.requests[rid]
            prompt_len = len(req.prompt_token_ids or [])
            # Emit one decode token once prompt is fully covered this step.
            if req.num_computed_tokens >= prompt_len:
                sampled.append([1])
            else:
                sampled.append([])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def _queue_ids(queue) -> list[str]:
        try:
            return [str(r.request_id) for r in queue]
        except TypeError:
            # Some queues are custom containers
            return [str(r.request_id) for r in list(queue)]

    for step in range(case.max_steps):
        while pending_idx < len(pending) and pending[pending_idx].arrive_step <= step:
            spec = pending[pending_idx]
            pending_idx += 1
            req = _build_request(spec)
            live[req.request_id] = req
            scheduler.add_request(req)

        if scheduler.get_num_unfinished_requests() == 0 and pending_idx >= len(pending):
            break

        # Snapshot preempted-in-waiting before schedule.
        waiting_before = set(_queue_ids(scheduler.waiting))
        preempted_before = {
            rid
            for rid in waiting_before
            if live.get(rid) is not None
            and getattr(live[rid], "status", None) == RequestStatus.PREEMPTED
        }

        output = scheduler.schedule()
        scheduled = {str(k): int(v) for k, v in output.num_scheduled_tokens.items()}
        new_ids = [str(r.req_id) for r in output.scheduled_new_reqs]

        # Prefer scheduler-reported preempt set (reset each schedule()).
        preempted_ids = sorted(
            str(x) for x in (getattr(output, "preempted_req_ids", None) or set())
        )
        if not preempted_ids:
            waiting_after = set(_queue_ids(scheduler.waiting))
            preempted_now = {
                rid
                for rid in waiting_after
                if rid in live
                and getattr(live[rid], "status", None) == RequestStatus.PREEMPTED
            }
            preempted_ids = sorted(preempted_now - preempted_before - prev_waiting_preempted)
            prev_waiting_preempted = preempted_now
        else:
            prev_waiting_preempted = set(preempted_ids)

        # Local APC hit for newly admitted requests:
        # after schedule(), num_computed_tokens == prefix_hit + scheduled.
        prefix_hits: dict[str, int] = {}
        if bool(s.get("enable_prefix_caching", False)):
            for rid in new_ids:
                req = live.get(rid)
                if req is None:
                    continue
                sched_n = int(scheduled.get(rid, 0))
                hit = max(0, int(req.num_computed_tokens) - sched_n)
                if hit > 0:
                    prefix_hits[rid] = hit

        finished_before = {
            rid for rid, req in live.items() if req.is_finished()
        }
        if scheduled:
            scheduler.update_from_output(output, _fake_output(output))
        finished_after = {
            rid for rid, req in live.items() if req.is_finished()
        }
        # Only newly finished this step (not cumulative scheduler set).
        finished_ids = sorted(finished_after - finished_before)

        free_blocks = int(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        allocated_blocks: dict[str, int] = {}
        for rid, req in live.items():
            if req.is_finished():
                continue
            try:
                blocks = scheduler.kv_cache_manager.get_blocks(rid)
                ids = blocks.get_block_ids()
                # get_block_ids returns tuple of lists (one per kv group)
                if ids and isinstance(ids[0], (list, tuple)):
                    n_blocks = len(ids[0])
                else:
                    n_blocks = len(ids) if ids else 0
            except Exception:
                n_blocks = 0
            if n_blocks > 0:
                allocated_blocks[str(rid)] = n_blocks

        records.append(
            ScheduleStepRecord(
                step=step,
                scheduled_tokens=scheduled,
                new_req_ids=sorted(new_ids),
                preempted_ids=preempted_ids,
                finished_ids=finished_ids,
                waiting_ids=_queue_ids(scheduler.waiting),
                running_ids=_queue_ids(scheduler.running),
                free_blocks=free_blocks,
                allocated_blocks=allocated_blocks,
                prefix_hit_tokens=prefix_hits,
            )
        )

        progressed = bool(scheduled or preempted_ids or finished_ids)
        arrived_now = any(p.arrive_step == step for p in case.requests)
        if progressed or arrived_now:
            idle_streak = 0
        else:
            idle_streak += 1
            if idle_streak >= 5 and pending_idx >= len(pending):
                break

        if scheduler.get_num_unfinished_requests() == 0 and pending_idx >= len(pending):
            break

    return records
