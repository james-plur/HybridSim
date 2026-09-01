"""Python bindings for the C++ flow-level network and topology plugins."""

from __future__ import annotations

import unittest

import hybridsim_py as hs

from hybridsim.network import (
    DirectTopology,
    FatTreeTopology,
    ShortestPathRouting,
    Topology,
    assemble_network,
    register_topology,
)


class TestNetworkPython(unittest.TestCase):
    def test_put_wait_via_engine(self) -> None:
        sim = hs.Simulation()
        net = assemble_network(
            sim,
            [(0, 0), (0, 1)],
            topology="fattree",
            routing="shortest_path",
            layers=1,
            link_bandwidth_bps=2000.0,
            link_delay_s=0.0,
            start=True,
        )

        src = hs.EngineActor(sim)
        dst = hs.EngineActor(sim)
        src.install_network(net, 0, 0)
        dst.install_network(net, 0, 1)
        done: dict[str, int] = {}
        src.set_on_workload_complete(lambda wid: done.update(src=int(wid)))
        dst.set_on_workload_complete(lambda wid: done.update(dst=int(wid)))
        src.start()
        dst.start()

        src.send_workload(
            {
                "workload_id": 1,
                "kernels": [
                    {
                        "name": "put",
                        "type": hs.KERNEL_PUT,
                        "params": {
                            "dst_addr": "0:1",
                            "conn_id": 11,
                            "payload_bytes": 200.0,
                        },
                    }
                ],
            }
        )
        dst.send_workload(
            {
                "workload_id": 2,
                "kernels": [
                    {
                        "name": "wait",
                        "type": hs.KERNEL_WAIT,
                        "params": {"conn_id": 11},
                    }
                ],
            }
        )
        sim.run()
        src.check_error()
        dst.check_error()
        net.rethrow_if_error()
        self.assertEqual(done["src"], 1)
        self.assertEqual(done["dst"], 2)
        self.assertAlmostEqual(sim.now(), 200.0 / 2000.0, places=9)

    def test_python_fattree_routes(self) -> None:
        sim = hs.Simulation()
        net = assemble_network(
            sim,
            [(0, 0), (0, 1)],
            layers=1,
            link_bandwidth_bps=1000.0,
            link_delay_s=0.0,
        )
        self.assertEqual(net.num_adapters(), 2)
        self.assertEqual(net.num_switches(), 1)
        ads = net.adapter_ids()
        sw = [i for i in net.node_ids() if not net.is_adapter(i)][0]
        self.assertEqual(net.nexthops(ads[0], 0, 1), [1])
        self.assertEqual(net.nexthops(sw, 0, 1), [1])
        self.assertEqual(net.nexthops(ads[1], 0, 0), [1])
        self.assertEqual(net.nexthops(sw, 0, 0), [0])

    def test_two_layer_python(self) -> None:
        sim = hs.Simulation()
        net = assemble_network(
            sim,
            [(0, 0), (1, 0)],
            layers=2,
            leaf_downlinks=1,
            leaf_uplinks=1,
            num_leaf=2,
            num_spine=1,
            link_bandwidth_bps=1000.0,
            link_delay_s=0.0,
            start=True,
        )
        self.assertEqual(net.num_switches(), 3)
        src = hs.EngineActor(sim)
        dst = hs.EngineActor(sim)
        src.install_network(net, 0, 0)
        dst.install_network(net, 1, 0)
        done: dict[str, int] = {}
        src.set_on_workload_complete(lambda wid: done.update(src=int(wid)))
        dst.set_on_workload_complete(lambda wid: done.update(dst=int(wid)))
        src.start()
        dst.start()
        src.send_workload(
            {
                "workload_id": 1,
                "kernels": [
                    {
                        "name": "put",
                        "type": hs.KERNEL_PUT,
                        "params": {
                            "dst_addr": "1:0",
                            "conn_id": 9,
                            "payload_bytes": 100.0,
                        },
                    }
                ],
            }
        )
        dst.send_workload(
            {
                "workload_id": 2,
                "kernels": [
                    {
                        "name": "wait",
                        "type": hs.KERNEL_WAIT,
                        "params": {"conn_id": 9},
                    }
                ],
            }
        )
        sim.run()
        src.check_error()
        dst.check_error()
        net.rethrow_if_error()
        self.assertEqual(done["src"], 1)
        self.assertEqual(done["dst"], 2)
        self.assertAlmostEqual(sim.now(), 100.0 / 1000.0, places=9)

    def test_custom_topology_plugin(self) -> None:
        @register_topology("line2")
        class Line2(Topology):
            def wire(self, net, addrs, *, bandwidth_bps, delay_s):
                endpoints = list(addrs)
                a = net.add_adapter(endpoints[0][0], endpoints[0][1], port_num=2)
                b = net.add_adapter(endpoints[1][0], endpoints[1][1], port_num=2)
                net.link(a, 1, b, 1, bandwidth_bps, delay_s)

        sim = hs.Simulation()
        net = assemble_network(
            sim,
            [(0, 0), (0, 1)],
            topology="line2",
            link_bandwidth_bps=4000.0,
            link_delay_s=0.0,
            start=True,
        )
        self.assertEqual(net.num_switches(), 0)
        self.assertEqual(net.num_adapters(), 2)
        src = hs.EngineActor(sim)
        dst = hs.EngineActor(sim)
        src.install_network(net, 0, 0)
        dst.install_network(net, 0, 1)
        done: dict[str, int] = {}
        src.set_on_workload_complete(lambda wid: done.update(src=int(wid)))
        dst.set_on_workload_complete(lambda wid: done.update(dst=int(wid)))
        src.start()
        dst.start()
        src.send_workload(
            {
                "workload_id": 1,
                "kernels": [
                    {
                        "name": "put",
                        "type": hs.KERNEL_PUT,
                        "params": {
                            "dst_addr": "0:1",
                            "conn_id": 1,
                            "payload_bytes": 40.0,
                        },
                    }
                ],
            }
        )
        dst.send_workload(
            {
                "workload_id": 2,
                "kernels": [
                    {
                        "name": "wait",
                        "type": hs.KERNEL_WAIT,
                        "params": {"conn_id": 1},
                    }
                ],
            }
        )
        sim.run()
        src.check_error()
        dst.check_error()
        net.rethrow_if_error()
        self.assertEqual(done["dst"], 2)
        self.assertAlmostEqual(sim.now(), 40.0 / 4000.0, places=9)

    def test_primitives_match_plugins(self) -> None:
        sim = hs.Simulation()
        net = hs.Network.create(sim)
        FatTreeTopology(layers=1).wire(
            net, [(0, 0), (0, 1)], bandwidth_bps=1000.0, delay_s=0.0
        )
        ShortestPathRouting().install(net)
        DirectTopology  # imported for registry
        self.assertEqual(net.node_count(), 3)
        peer = net.downstream(0, 1)
        self.assertIsNotNone(peer)
        self.assertEqual(peer[0], 2)


if __name__ == "__main__":
    unittest.main()
