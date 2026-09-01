"""Flow-level fabric simulation (optional; default off keeps α-β TimeoutKernels)."""

from __future__ import annotations

from dataclasses import dataclass


_VALID_BW = frozenset({"max_min", "ingress_proportional", "priority_then_maxmin"})
_VALID_LB = frozenset({"ecmp_hash", "random", "least_loaded"})


@dataclass
class NetworkSimConfig:
    """C++ flow-level network. Topology and routing tables are built in Python.

    Distinct from ``NetworkConfig`` (α-β formula used when this is disabled).
    """

    enabled: bool = False
    #: Wire plugin: ``fattree`` or ``direct``.
    topology: str = "fattree"
    #: Route plugin: ``shortest_path`` (all equal-cost next hops).
    routing: str = "shortest_path"
    #: FatTree depth: ``1`` (single switch) or ``2`` (leaf-spine).
    layers: int = 1
    num_leaf: int = 0
    num_spine: int = 0
    leaf_downlinks: int = 0
    leaf_uplinks: int = 0
    #: Wire capacity (bytes/s). Default ~50 Gbps.
    link_bandwidth_bps: float = 50e9 / 8.0
    link_delay_s: float = 1e-6
    #: ``max_min`` / ``ingress_proportional`` / ``priority_then_maxmin``.
    bw_policy: str = "max_min"
    #: ``ecmp_hash`` / ``random`` / ``least_loaded``.
    lb_policy: str = "ecmp_hash"
    seed: int = 0
    #: 0 → ``max(attn_tp, moe_tp, ep_size)``.
    ranks_per_replica: int = 0

    def resolved_bw_policy(self) -> str:
        p = (self.bw_policy or "").lower().strip()
        if p not in _VALID_BW:
            raise ValueError(
                "network_sim.bw_policy must be one of "
                f"{sorted(_VALID_BW)}, got {self.bw_policy!r}"
            )
        return p

    def resolved_lb_policy(self) -> str:
        p = (self.lb_policy or "").lower().strip()
        if p not in _VALID_LB:
            raise ValueError(
                "network_sim.lb_policy must be one of "
                f"{sorted(_VALID_LB)}, got {self.lb_policy!r}"
            )
        return p

    def resolved_topology(self) -> str:
        p = (self.topology or "fattree").lower().strip()
        if not p:
            raise ValueError("network_sim.topology is empty")
        return p

    def resolved_routing(self) -> str:
        p = (self.routing or "shortest_path").lower().strip()
        if not p:
            raise ValueError("network_sim.routing is empty")
        return p

    def resolved_layers(self) -> int:
        layers = int(self.layers)
        if layers not in (1, 2):
            raise ValueError("network_sim.layers must be 1 or 2")
        return layers

    def resolved_ranks(self, parallel: object | None) -> int:
        from hybridsim_infer.workload_generators.infer_workload_generator.op_level.comm import (
            ranks_per_replica,
        )

        return ranks_per_replica(parallel, self.ranks_per_replica)
