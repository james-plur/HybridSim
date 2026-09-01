"""Cluster dispatch + replica scheduler + worker engine knobs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterScheduleConfig:
    """How ClusterActor picks a replica. Only ``least_load`` is implemented."""

    policy: str = "least_load"


@dataclass
class ReplicaScheduleConfig:
    """Replica-local scheduler (``SchedulerFactory`` name + vLLM knobs)."""

    #: Schedule backend: ``vllm`` (more via ``SchedulerFactory.register``).
    name: str = "vllm"
    #: Cap on tokens scheduled for one request's prefill chunk in a step.
    tokens_per_step: int = 8
    #: Decode tokens scheduled per request per step (vLLM-like default: 1).
    decode_tokens_per_step: int = 1
    #: vLLM-style per-step token budget across all requests (0 → unlimited).
    max_num_scheduled_tokens: int = 64
    #: Max concurrent running requests.
    max_num_running_reqs: int = 32
    #: Long-prefill threshold; 0 means use ``tokens_per_step``.
    long_prefill_token_threshold: int = 0
    #: Match vLLM ``scheduler_reserve_full_isl``: admit only if full sequence fits.
    reserve_full_isl: bool = True


@dataclass
class EngineConfig:
    """WorkerEngine pipeline (not a VllmScheduler field)."""

    #: Max concurrent Worker batches per replica. Occupancy held until BatchEnd.
    max_inflight_batches: int = 1


@dataclass
class ScheduleConfig:
    cluster: ClusterScheduleConfig = field(default_factory=ClusterScheduleConfig)
    replica: ReplicaScheduleConfig = field(default_factory=ReplicaScheduleConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
