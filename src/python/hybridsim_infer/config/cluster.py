"""Cluster topology (replica pools). Dispatch policy lives under schedule."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_TYPES = frozenset({"monolith", "pd"})


@dataclass
class ClusterConfig:
    """Serving topology: monolith replicas or Prefill/Decode pools."""

    #: ``monolith`` (all replicas equal) or ``pd`` (Prefill + Decode pools).
    type: str = "monolith"
    #: Monolith: total replicas. Ignored when ``type=pd`` (derived from P+D).
    num_replicas: int = 1
    #: PD: Prefill-pool replicas (ids ``0 .. Np-1``).
    num_prefill_replicas: int = 1
    #: PD: Decode-pool replicas (ids ``Np .. Np+Nd-1``).
    num_decode_replicas: int = 1

    def resolved_cluster_type(self) -> str:
        ct = (self.type or "").lower().strip()
        if ct not in _VALID_TYPES:
            raise ValueError(
                f"cluster.type must be 'monolith' or 'pd', got {self.type!r}"
            )
        return ct

    def resolved_num_replicas(self) -> int:
        if self.resolved_cluster_type() == "pd":
            return int(self.num_prefill_replicas) + int(self.num_decode_replicas)
        return int(self.num_replicas)

    def pd_pools(self) -> tuple[list[int], list[int]]:
        np_ = int(self.num_prefill_replicas)
        nd_ = int(self.num_decode_replicas)
        prefill = list(range(np_))
        decode = list(range(np_, np_ + nd_))
        return prefill, decode
