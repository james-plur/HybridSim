#include "bindings_common.hpp"
#include "network_bindings.hpp"

#include "hybridsim/network.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace hybridsim::python {

void bind_network(py::module_ &m) {
  namespace net = hybridsim::network;

  py::enum_<net::BwPolicyKind>(m, "BwPolicyKind")
      .value("max_min", net::BwPolicyKind::MaxMin)
      .value("ingress_proportional", net::BwPolicyKind::IngressProportional)
      .value("priority_then_maxmin", net::BwPolicyKind::PriorityThenMaxMin);

  py::enum_<net::LbPolicyKind>(m, "LbPolicyKind")
      .value("ecmp_hash", net::LbPolicyKind::EcmpHash)
      .value("random", net::LbPolicyKind::Random)
      .value("least_loaded", net::LbPolicyKind::LeastLoaded);

  py::class_<net::NetworkAddr>(m, "NetworkAddr")
      .def(py::init<>())
      .def(py::init<int32_t, int32_t>(), py::arg("replica_id") = 0,
           py::arg("rank") = 0)
      .def_readwrite("replica_id", &net::NetworkAddr::replica_id)
      .def_readwrite("rank", &net::NetworkAddr::rank)
      .def("to_string", &net::NetworkAddr::to_string)
      .def_static("parse", &net::NetworkAddr::parse, py::arg("text"))
      .def("__repr__", [](const net::NetworkAddr &self) {
        return "<NetworkAddr " + self.to_string() + ">";
      });

  py::class_<net::NetworkBuildConfig>(m, "NetworkBuildConfig")
      .def(py::init<>())
      .def_readwrite("layers", &net::NetworkBuildConfig::layers)
      .def_readwrite("num_leaf", &net::NetworkBuildConfig::num_leaf)
      .def_readwrite("num_spine", &net::NetworkBuildConfig::num_spine)
      .def_readwrite("leaf_downlinks", &net::NetworkBuildConfig::leaf_downlinks)
      .def_readwrite("leaf_uplinks", &net::NetworkBuildConfig::leaf_uplinks)
      .def_readwrite("link_bandwidth_bps",
                     &net::NetworkBuildConfig::link_bandwidth_bps)
      .def_readwrite("link_delay_s", &net::NetworkBuildConfig::link_delay_s)
      .def_readwrite("bw_policy", &net::NetworkBuildConfig::bw_policy)
      .def_readwrite("lb_policy", &net::NetworkBuildConfig::lb_policy)
      .def_readwrite("seed", &net::NetworkBuildConfig::seed);

  py::class_<net::Network, std::shared_ptr<net::Network>>(m, "Network")
      .def(py::init([](std::shared_ptr<SimulationState> state,
                       net::BwPolicyKind bw, net::LbPolicyKind lb,
                       uint32_t seed) {
             if (!state || !state->sim) {
               throw std::runtime_error("Network requires a Simulation");
             }
             return net::Network::create(*state->sim, bw, lb, seed);
           }),
           py::arg("sim"),
           py::arg("bw_policy") = net::BwPolicyKind::MaxMin,
           py::arg("lb_policy") = net::LbPolicyKind::EcmpHash,
           py::arg("seed") = 0, py::keep_alive<1, 2>())
      .def_static(
          "create",
          [](std::shared_ptr<SimulationState> state, net::BwPolicyKind bw,
             net::LbPolicyKind lb, uint32_t seed) {
            if (!state || !state->sim) {
              throw std::runtime_error("Network.create requires a Simulation");
            }
            return net::Network::create(*state->sim, bw, lb, seed);
          },
          py::arg("sim"), py::arg("bw_policy") = net::BwPolicyKind::MaxMin,
          py::arg("lb_policy") = net::LbPolicyKind::EcmpHash,
          py::arg("seed") = 0, py::keep_alive<0, 1>())
      .def_static(
          "build",
          [](std::shared_ptr<SimulationState> state,
             const net::NetworkBuildConfig &cfg,
             const std::vector<std::pair<int32_t, int32_t>> &addr_pairs) {
            if (!state || !state->sim) {
              throw std::runtime_error("Network.build requires a Simulation");
            }
            std::vector<net::NetworkAddr> addrs;
            addrs.reserve(addr_pairs.size());
            for (const auto &[replica, rank] : addr_pairs) {
              addrs.push_back(net::NetworkAddr{replica, rank});
            }
            return net::Network::build(*state->sim, cfg, std::move(addrs));
          },
          py::arg("sim"), py::arg("config"), py::arg("addrs"),
          py::keep_alive<0, 1>())
      .def(
          "add_adapter",
          [](net::Network &self, int32_t replica_id, int32_t rank,
             int port_num) {
            return self.add_adapter(net::NetworkAddr{replica_id, rank},
                                    port_num);
          },
          py::arg("replica_id"), py::arg("rank"), py::arg("port_num") = 2,
          "Create an endpoint adapter; returns node id")
      .def(
          "add_switch", &net::Network::add_switch, py::arg("port_num"),
          "Create a switch; returns node id")
      .def("link", &net::Network::link, py::arg("a"), py::arg("a_port"),
           py::arg("b"), py::arg("b_port"), py::arg("bandwidth_bps"),
           py::arg("delay_s"),
           "Bidirectional link A.out[a_port] ↔ B.in[b_port]")
      .def(
          "set_nexthops",
          [](net::Network &self, int node_id, int32_t replica_id, int32_t rank,
             const std::vector<int> &ports) {
            self.set_nexthops(node_id, net::NetworkAddr{replica_id, rank},
                              ports);
          },
          py::arg("node_id"), py::arg("dst_replica"), py::arg("dst_rank"),
          py::arg("ports"),
          "Install equal-cost next-hop out-ports toward dst_replica:dst_rank")
      .def(
          "nexthops",
          [](const net::Network &self, int node_id, int32_t replica_id,
             int32_t rank) {
            return self.nexthops(node_id, net::NetworkAddr{replica_id, rank});
          },
          py::arg("node_id"), py::arg("dst_replica"), py::arg("dst_rank"))
      .def("node_count", &net::Network::node_count)
      .def("port_num", &net::Network::port_num, py::arg("node_id"))
      .def("is_adapter", &net::Network::is_adapter, py::arg("node_id"))
      .def(
          "node_addr",
          [](const net::Network &self, int node_id) {
            return self.node_addr(node_id);
          },
          py::arg("node_id"))
      .def(
          "downstream",
          [](const net::Network &self, int node_id, int out_port)
              -> py::object {
            const auto peer = self.downstream(node_id, out_port);
            if (!peer) {
              return py::none();
            }
            return py::make_tuple(peer->first, peer->second);
          },
          py::arg("node_id"), py::arg("out_port"),
          "Peer (node_id, in_port) of this out port, or None")
      .def("adapter_ids", &net::Network::adapter_ids)
      .def("node_ids", &net::Network::node_ids)
      .def("install_shortest_path_routes",
           &net::Network::install_shortest_path_routes,
           "C++ helper: fill routing tables from current links")
      .def("start", &net::Network::start)
      .def("stop", &net::Network::stop)
      .def("num_adapters", &net::Network::num_adapters)
      .def("num_switches", &net::Network::num_switches)
      .def("addrs", &net::Network::addrs)
      .def("rethrow_if_error", &net::Network::rethrow_if_error);

  m.attr("KERNEL_TIMEOUT") = py::int_(0);
  m.attr("KERNEL_PUT") = py::int_(1);
  m.attr("KERNEL_SIGNAL") = py::int_(2);
  m.attr("KERNEL_WAIT") = py::int_(3);
  m.attr("KERNEL_GET") = py::int_(4);
}

} // namespace hybridsim::python
