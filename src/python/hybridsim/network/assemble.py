"""Assemble a C++ Network from Python topology + routing plugins."""

from __future__ import annotations

import inspect
from typing import Any, Sequence

from hybridsim.network.routing import get_routing
from hybridsim.network.topology import get_topology


def _hs_module():
    import hybridsim_py as hs

    return hs


def _policy_enums(hs, bw_policy: str, lb_policy: str):
    bw_key = str(bw_policy).lower().strip()
    lb_key = str(lb_policy).lower().strip()
    bw = getattr(hs.BwPolicyKind, bw_key, None)
    lb = getattr(hs.LbPolicyKind, lb_key, None)
    if bw is None:
        raise ValueError(f"unknown bw_policy {bw_policy!r}")
    if lb is None:
        raise ValueError(f"unknown lb_policy {lb_policy!r}")
    return bw, lb


def _init_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def assemble_network(
    hs_sim: Any,
    addrs: Sequence[tuple[int, int]],
    *,
    topology: str = "fattree",
    routing: str = "shortest_path",
    bw_policy: str = "max_min",
    lb_policy: str = "ecmp_hash",
    seed: int = 0,
    link_bandwidth_bps: float = 50e9 / 8.0,
    link_delay_s: float = 1e-6,
    start: bool = False,
    **topo_kwargs: Any,
) -> Any:
    """Create adapters/switches in C++, wire and route them in Python.

    ``topo_kwargs`` are forwarded to the topology constructor (e.g. FatTree
    ``layers`` / ``num_leaf`` / ``leaf_downlinks``). Unknown keys are ignored.
    """
    hs = _hs_module()
    bw, lb = _policy_enums(hs, bw_policy, lb_policy)
    net = hs.Network.create(hs_sim, bw, lb, int(seed))
    topo_cls = get_topology(topology)
    topo_cls(**_init_kwargs(topo_cls, topo_kwargs)).wire(
        net,
        [(int(r), int(k)) for r, k in addrs],
        bandwidth_bps=float(link_bandwidth_bps),
        delay_s=float(link_delay_s),
    )
    get_routing(routing)().install(net)
    if start:
        net.start()
    return net
