"""Routing table plugins. Default is all equal-cost shortest next hops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any


class Routing(ABC):
    """Fill ``net`` routing tables from the current link graph."""

    @abstractmethod
    def install(self, net: Any) -> None:
        """Write next-hop out-ports on every node."""


_ROUTINGS: dict[str, type[Routing]] = {}


def register_routing(name: str):
    def deco(cls: type[Routing]) -> type[Routing]:
        _ROUTINGS[str(name).lower().strip()] = cls
        return cls

    return deco


def get_routing(name: str) -> type[Routing]:
    key = str(name).lower().strip()
    if key not in _ROUTINGS:
        raise ValueError(
            f"unknown routing {name!r}; registered: {sorted(_ROUTINGS)}"
        )
    return _ROUTINGS[key]


def list_routings() -> list[str]:
    return sorted(_ROUTINGS)


@register_routing("shortest_path")
class ShortestPathRouting(Routing):
    """BFS from each adapter; keep every equal-cost next hop (ECMP)."""

    def install(self, net: Any) -> None:
        n = int(net.node_count())
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for u in range(n):
            for p in range(int(net.port_num(u))):
                peer = net.downstream(u, p)
                if peer is None:
                    continue
                v, _in_port = peer
                adj[u].append((int(p), int(v)))

        for dest in net.adapter_ids():
            dest = int(dest)
            addr = net.node_addr(dest)
            dist = [-1] * n
            dist[dest] = 0
            q: deque[int] = deque([dest])
            while q:
                u = q.popleft()
                for _port, v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for u in range(n):
                if u == dest or dist[u] < 0:
                    continue
                want = dist[u] - 1
                hops = [port for port, v in adj[u] if dist[v] == want]
                if hops:
                    net.set_nexthops(
                        u, int(addr.replica_id), int(addr.rank), hops
                    )
