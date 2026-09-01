#pragma once

#include "hybridsim/network/topology.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hybridsim::network {

class Network {
public:
  Network(simcpp20::simulation<> &sim, BwPolicyKind bw = BwPolicyKind::MaxMin,
          LbPolicyKind lb = LbPolicyKind::EcmpHash, uint32_t seed = 0)
      : sim_(&sim), bw_(bw), lb_(lb), seed_(seed) {}

  static std::shared_ptr<Network> create(simcpp20::simulation<> &sim,
                                         BwPolicyKind bw = BwPolicyKind::MaxMin,
                                         LbPolicyKind lb = LbPolicyKind::EcmpHash,
                                         uint32_t seed = 0) {
    return std::shared_ptr<Network>(new Network(sim, bw, lb, seed));
  }

  /// C++ test helper: FatTree wiring + shortest-path routes.
  static std::shared_ptr<Network> build(simcpp20::simulation<> &sim,
                                        const NetworkBuildConfig &cfg,
                                        std::vector<NetworkAddr> addrs) {
    auto net = create(sim, cfg.bw_policy, cfg.lb_policy, cfg.seed);
    net->cfg_ = cfg;
    net->adopt(build_fattree(sim, cfg, std::move(addrs)));
    return net;
  }

  ~Network() {
    if (started_) {
      stop();
    }
  }

  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;

  int add_adapter(const NetworkAddr &addr, int port_num = 2) {
    if (port_num < 1) {
      throw std::invalid_argument("adapter needs at least 1 port");
    }
    if (by_addr_.count(addr.key()) != 0) {
      throw std::invalid_argument("duplicate adapter addr " + addr.to_string());
    }
    const int adapter_index = static_cast<int>(adapters_.size());
    auto adapter = std::make_unique<NetworkAdapter>(
        *sim_, addr, port_num, adapter_index, bw_, lb_,
        seed_ + static_cast<uint32_t>(adapter_index));
    const int id = static_cast<int>(nodes_.size());
    id_of_[adapter.get()] = id;
    by_addr_[addr.key()] = adapter.get();
    nodes_.push_back(adapter.get());
    adapters_.push_back(std::move(adapter));
    return id;
  }

  int add_switch(int port_num) {
    if (port_num < 1) {
      throw std::invalid_argument("switch needs at least 1 port");
    }
    const int sw_index = static_cast<int>(switches_.size());
    auto sw = std::make_unique<NetworkSwitch>(
        *sim_, port_num, sw_index, bw_, lb_,
        seed_ + 1000u + static_cast<uint32_t>(sw_index));
    const int id = static_cast<int>(nodes_.size());
    id_of_[sw.get()] = id;
    nodes_.push_back(sw.get());
    switches_.push_back(std::move(sw));
    return id;
  }

  void link(int a_id, int a_port, int b_id, int b_port, double bandwidth_bps,
            double delay_s) {
    Node &a = *require_node(a_id);
    Node &b = *require_node(b_id);
    if (a_port < 0 || a_port >= a.port_num() || b_port < 0 ||
        b_port >= b.port_num()) {
      throw std::out_of_range("link port out of range");
    }
    if (bandwidth_bps <= 0.0) {
      throw std::invalid_argument("link bandwidth_bps must be positive");
    }
    link_bidirectional(a, a_port, b, b_port, bandwidth_bps, delay_s);
  }

  void set_nexthops(int node_id, const NetworkAddr &dst,
                    std::vector<int> ports) {
    Node &node = *require_node(node_id);
    for (int p : ports) {
      if (p < 0 || p >= node.port_num()) {
        throw std::out_of_range("nexthop port out of range");
      }
    }
    node.router().set_nexthops(dst, std::move(ports));
  }

  std::vector<int> nexthops(int node_id, const NetworkAddr &dst) const {
    const Node &node = *require_node(node_id);
    const auto *hops = node.router().nexthops(dst);
    if (hops == nullptr) {
      return {};
    }
    return *hops;
  }

  int node_count() const noexcept { return static_cast<int>(nodes_.size()); }

  int port_num(int node_id) const { return require_node(node_id)->port_num(); }

  bool is_adapter(int node_id) const {
    return require_node(node_id)->is_endpoint();
  }

  NetworkAddr node_addr(int node_id) const {
    return require_node(node_id)->addr();
  }

  std::optional<std::pair<int, int>> downstream(int node_id,
                                                int out_port) const {
    Node &node = *require_node(node_id);
    if (out_port < 0 || out_port >= node.port_num()) {
      throw std::out_of_range("out_port out of range");
    }
    InPort *peer = node.out_port(out_port).downstream();
    if (peer == nullptr) {
      return std::nullopt;
    }
    const auto it = id_of_.find(&peer->owner());
    if (it == id_of_.end()) {
      return std::nullopt;
    }
    return std::make_pair(it->second, peer->index());
  }

  std::vector<int> adapter_ids() const {
    std::vector<int> out;
    out.reserve(adapters_.size());
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
      if (nodes_[static_cast<std::size_t>(i)]->is_endpoint()) {
        out.push_back(i);
      }
    }
    return out;
  }

  std::vector<int> node_ids() const {
    std::vector<int> out(nodes_.size());
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
      out[static_cast<std::size_t>(i)] = i;
    }
    return out;
  }

  void install_shortest_path_routes() {
    std::vector<NetworkAdapter *> adapter_ptrs;
    adapter_ptrs.reserve(adapters_.size());
    for (auto &a : adapters_) {
      adapter_ptrs.push_back(a.get());
    }
    compute_routing_tables(nodes_, adapter_ptrs);
  }

  void start() {
    if (started_) {
      return;
    }
    for (auto &a : adapters_) {
      a->start_ports();
    }
    for (auto &s : switches_) {
      s->start_ports();
    }
    started_ = true;
  }

  void stop() {
    for (auto &a : adapters_) {
      a->stop_ports();
    }
    for (auto &s : switches_) {
      s->stop_ports();
    }
    started_ = false;
  }

  void rethrow_if_error() const {
    for (const auto &a : adapters_) {
      a->rethrow_port_errors();
    }
    for (const auto &s : switches_) {
      s->rethrow_port_errors();
    }
  }

  NetworkAdapter *adapter(const NetworkAddr &addr) const {
    const auto it = by_addr_.find(addr.key());
    if (it == by_addr_.end()) {
      throw std::out_of_range("no adapter for addr " + addr.to_string());
    }
    return it->second;
  }

  NetworkAdapter *adapter(int32_t replica_id, int32_t rank) const {
    return adapter(NetworkAddr{replica_id, rank});
  }

  std::size_t num_adapters() const noexcept { return adapters_.size(); }
  std::size_t num_switches() const noexcept { return switches_.size(); }

  const NetworkBuildConfig &config() const noexcept { return cfg_; }

  BwPolicyKind bw_policy() const noexcept { return bw_; }
  LbPolicyKind lb_policy() const noexcept { return lb_; }

  std::vector<NetworkAddr> addrs() const {
    std::vector<NetworkAddr> out;
    out.reserve(adapters_.size());
    for (const auto &a : adapters_) {
      out.push_back(a->addr());
    }
    return out;
  }

private:
  void adopt(TopologyBuild &&topo) {
    adapters_.clear();
    switches_.clear();
    nodes_.clear();
    id_of_.clear();
    by_addr_.clear();
    for (auto &adapter : topo.adapters) {
      const int id = static_cast<int>(nodes_.size());
      id_of_[adapter.get()] = id;
      by_addr_[adapter->addr().key()] = adapter.get();
      nodes_.push_back(adapter.get());
      adapters_.push_back(std::move(adapter));
    }
    for (auto &sw : topo.switches) {
      const int id = static_cast<int>(nodes_.size());
      id_of_[sw.get()] = id;
      nodes_.push_back(sw.get());
      switches_.push_back(std::move(sw));
    }
  }

  Node *require_node(int node_id) const {
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size())) {
      throw std::out_of_range("unknown network node id");
    }
    return nodes_[static_cast<std::size_t>(node_id)];
  }

  simcpp20::simulation<> *sim_ = nullptr;
  BwPolicyKind bw_ = BwPolicyKind::MaxMin;
  LbPolicyKind lb_ = LbPolicyKind::EcmpHash;
  uint32_t seed_ = 0;
  NetworkBuildConfig cfg_{};
  std::vector<std::unique_ptr<NetworkAdapter>> adapters_;
  std::vector<std::unique_ptr<NetworkSwitch>> switches_;
  std::vector<Node *> nodes_;
  std::unordered_map<Node *, int> id_of_;
  std::unordered_map<uint64_t, NetworkAdapter *> by_addr_;
  bool started_ = false;
};

} // namespace hybridsim::network
