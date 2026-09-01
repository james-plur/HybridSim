"""Python topology plugins: wire adapters/switches, then install routes.

C++ exposes node/link/nexthop primitives. Subclass ``Topology`` / ``Routing``
and ``register_*`` to add a new fabric without changing the data plane.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

Addr = tuple[int, int]


class Topology(ABC):
    """Create adapters/switches and link ports. Does not install routes."""

    @abstractmethod
    def wire(
        self,
        net: Any,
        addrs: Sequence[Addr],
        *,
        bandwidth_bps: float,
        delay_s: float,
    ) -> None:
        """Populate ``net`` (a ``hybridsim_py.Network``)."""


_TOPOLOGIES: dict[str, type[Topology]] = {}


def register_topology(name: str):
    """Decorator: ``@register_topology("fattree")``."""

    def deco(cls: type[Topology]) -> type[Topology]:
        key = str(name).lower().strip()
        _TOPOLOGIES[key] = cls
        return cls

    return deco


def get_topology(name: str) -> type[Topology]:
    key = str(name).lower().strip()
    if key not in _TOPOLOGIES:
        raise ValueError(
            f"unknown topology {name!r}; registered: {sorted(_TOPOLOGIES)}"
        )
    return _TOPOLOGIES[key]


def list_topologies() -> list[str]:
    return sorted(_TOPOLOGIES)


@register_topology("fattree")
class FatTreeTopology(Topology):
    """1-layer star or 2-layer leaf–spine. Zero sizes are derived from endpoints."""

    def __init__(
        self,
        *,
        layers: int = 1,
        num_leaf: int = 0,
        num_spine: int = 0,
        leaf_downlinks: int = 0,
        leaf_uplinks: int = 0,
    ) -> None:
        self.layers = int(layers)
        self.num_leaf = int(num_leaf)
        self.num_spine = int(num_spine)
        self.leaf_downlinks = int(leaf_downlinks)
        self.leaf_uplinks = int(leaf_uplinks)

    def wire(
        self,
        net: Any,
        addrs: Sequence[Addr],
        *,
        bandwidth_bps: float,
        delay_s: float,
    ) -> None:
        endpoints = [(int(r), int(k)) for r, k in addrs]
        if not endpoints:
            raise ValueError("fattree requires at least one endpoint")
        if self.layers not in (1, 2):
            raise ValueError("fattree layers must be 1 or 2")
        bw = float(bandwidth_bps)
        delay = float(delay_s)
        n = len(endpoints)
        adapters = [
            net.add_adapter(rep, rank, port_num=2) for rep, rank in endpoints
        ]
        if self.layers == 1:
            sw = net.add_switch(n)
            for i, ad in enumerate(adapters):
                net.link(ad, 1, sw, i, bw, delay)
            return
        self._wire_two_layer(net, adapters, n, bw, delay)

    def _wire_two_layer(
        self,
        net: Any,
        adapters: list[int],
        n: int,
        bw: float,
        delay: float,
    ) -> None:
        leaf_down = self.leaf_downlinks
        if leaf_down <= 0:
            leaf_down = max(1, min(4, n))
        num_leaf = self.num_leaf
        if num_leaf <= 0:
            num_leaf = (n + leaf_down - 1) // leaf_down
        if num_leaf * leaf_down < n:
            raise ValueError(
                "fattree 2-layer: num_leaf * leaf_downlinks < endpoints"
            )
        leaf_up = self.leaf_uplinks if self.leaf_uplinks > 0 else 1
        num_spine = self.num_spine if self.num_spine > 0 else max(1, leaf_up)
        if leaf_up > num_spine:
            num_spine = leaf_up

        leaf_ports = leaf_down + leaf_up
        leaves = [net.add_switch(leaf_ports) for _ in range(num_leaf)]
        spines = [net.add_switch(num_leaf) for _ in range(num_spine)]
        for i, ad in enumerate(adapters):
            leaf_id = i // leaf_down
            down_port = i % leaf_down
            net.link(ad, 1, leaves[leaf_id], down_port, bw, delay)
        for li, leaf in enumerate(leaves):
            for u in range(leaf_up):
                spine_id = u % num_spine
                leaf_up_port = leaf_down + u
                net.link(leaf, leaf_up_port, spines[spine_id], li, bw, delay)


@register_topology("direct")
class DirectTopology(Topology):
    """Pairwise adapter uplinks with no switch (2 endpoints, or full mesh)."""

    def wire(
        self,
        net: Any,
        addrs: Sequence[Addr],
        *,
        bandwidth_bps: float,
        delay_s: float,
    ) -> None:
        endpoints = [(int(r), int(k)) for r, k in addrs]
        n = len(endpoints)
        if n < 2:
            raise ValueError("direct topology needs at least 2 endpoints")
        # port 0 = host; 1 .. n-1 = peer links
        adapters = [
            net.add_adapter(rep, rank, port_num=n) for rep, rank in endpoints
        ]
        bw = float(bandwidth_bps)
        delay = float(delay_s)
        for i in range(n):
            for j in range(i + 1, n):
                # port 0 is host; peer k uses port k+1 if k < i else k
                port_i = j  # j > i → port j
                port_j = i + 1  # i < j → port i+1
                net.link(adapters[i], port_i, adapters[j], port_j, bw, delay)
