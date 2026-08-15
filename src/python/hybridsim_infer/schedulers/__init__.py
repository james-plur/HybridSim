"""Pluggable replica schedulers (schedule + batch completion)."""

from hybridsim_infer.schedulers.factory import (
    InferenceScheduler,
    RemoteLookupFn,
    SchedulerFactory,
)
from hybridsim_infer.schedulers.vllm_schedule import VllmScheduler

SchedulerFactory.register("vllm", VllmScheduler)

__all__ = [
    "InferenceScheduler",
    "RemoteLookupFn",
    "SchedulerFactory",
    "VllmScheduler",
]
