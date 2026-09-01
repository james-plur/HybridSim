"""Deprecated import path; use ``op_level.comm`` (``RingCommAnalyzer``)."""

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.comm import (
    KERNEL_GET,
    KERNEL_PUT,
    KERNEL_SIGNAL,
    KERNEL_TIMEOUT,
    KERNEL_WAIT,
    RingCommAnalyzer,
    RingCommParser,
    addr_str,
    encode_conn,
    ranks_per_replica,
)

__all__ = [
    "KERNEL_GET",
    "KERNEL_PUT",
    "KERNEL_SIGNAL",
    "KERNEL_TIMEOUT",
    "KERNEL_WAIT",
    "RingCommAnalyzer",
    "RingCommParser",
    "addr_str",
    "encode_conn",
    "ranks_per_replica",
]
