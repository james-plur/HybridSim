"""Python-side network topology and routing (C++ fabric underneath)."""

from hybridsim.network.assemble import assemble_network
from hybridsim.network.routing import (
    Routing,
    ShortestPathRouting,
    get_routing,
    list_routings,
    register_routing,
)
from hybridsim.network.topology import (
    DirectTopology,
    FatTreeTopology,
    Topology,
    get_topology,
    list_topologies,
    register_topology,
)

__all__ = [
    "DirectTopology",
    "FatTreeTopology",
    "Routing",
    "ShortestPathRouting",
    "Topology",
    "assemble_network",
    "get_routing",
    "get_topology",
    "list_routings",
    "list_topologies",
    "register_routing",
    "register_topology",
]
