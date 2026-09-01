#pragma once

#include "hybridsim/network/adapter.hpp"
#include "hybridsim/network/types.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hybridsim::network {

inline void link_ports(OutPort &a, InPort &b, double bandwidth_bps,
                       double delay_s) {
  a.set_downstream(&b, bandwidth_bps, delay_s);
}

inline void link_bidirectional(Node &a, int a_port, Node &b, int b_port,
                               double bandwidth_bps, double delay_s) {
  link_ports(a.out_port(a_port), b.in_port(b_port), bandwidth_bps, delay_s);
  link_ports(b.out_port(b_port), a.in_port(a_port), bandwidth_bps, delay_s);
}

struct TopologyBuild {
  std::vector<std::unique_ptr<NetworkAdapter>> adapters;
  std::vector<std::unique_ptr<NetworkSwitch>> switches;
};

inline void compute_routing_tables(const std::vector<Node *> &nodes,
                                   const std::vector<NetworkAdapter *> &adapters) {
  const int n = static_cast<int>(nodes.size());
  std::unordered_map<Node *, int> index;
  index.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    index[nodes[static_cast<std::size_t>(i)]] = i;
  }

  std::vector<std::vector<std::pair<int, int>>> adj(
      static_cast<std::size_t>(n)); // (out_port, neighbor_index)
  for (int i = 0; i < n; ++i) {
    Node *node = nodes[static_cast<std::size_t>(i)];
    for (int p = 0; p < node->port_num(); ++p) {
      InPort *peer = node->out_port(p).downstream();
      if (peer == nullptr) {
        continue;
      }
      Node *other = &peer->owner();
      const auto it = index.find(other);
      if (it == index.end()) {
        continue;
      }
      adj[static_cast<std::size_t>(i)].push_back({p, it->second});
    }
  }

  for (NetworkAdapter *dest : adapters) {
    const int d = index[dest];
    std::vector<int> dist(static_cast<std::size_t>(n), -1);
    std::queue<int> q;
    dist[static_cast<std::size_t>(d)] = 0;
    q.push(d);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      for (const auto &[port, v] : adj[static_cast<std::size_t>(u)]) {
        (void)port;
        if (dist[static_cast<std::size_t>(v)] >= 0) {
          continue;
        }
        dist[static_cast<std::size_t>(v)] =
            dist[static_cast<std::size_t>(u)] + 1;
        q.push(v);
      }
    }
    for (int u = 0; u < n; ++u) {
      if (u == d || dist[static_cast<std::size_t>(u)] < 0) {
        continue;
      }
      std::vector<int> hops;
      const int want = dist[static_cast<std::size_t>(u)] - 1;
      for (const auto &[port, v] : adj[static_cast<std::size_t>(u)]) {
        if (dist[static_cast<std::size_t>(v)] == want) {
          hops.push_back(port);
        }
      }
      if (!hops.empty()) {
        nodes[static_cast<std::size_t>(u)]->router().set_nexthops(dest->addr(),
                                                                  std::move(hops));
      }
    }
  }
}

inline TopologyBuild build_fattree(simcpp20::simulation<> &sim,
                                   const NetworkBuildConfig &cfg,
                                   const std::vector<NetworkAddr> &addrs) {
  if (addrs.empty()) {
    throw std::invalid_argument("fattree requires at least one endpoint");
  }
  if (cfg.layers != 1 && cfg.layers != 2) {
    throw std::invalid_argument("fattree layers must be 1 or 2");
  }
  if (cfg.link_bandwidth_bps <= 0.0) {
    throw std::invalid_argument("link_bandwidth_bps must be positive");
  }

  TopologyBuild topo;
  const int n = static_cast<int>(addrs.size());
  const int adapter_ports = 2; // 0 host, 1 network uplink

  auto make_adapter = [&](int i) {
    return std::make_unique<NetworkAdapter>(
        sim, addrs[static_cast<std::size_t>(i)], adapter_ports, i,
        cfg.bw_policy, cfg.lb_policy, cfg.seed + static_cast<uint32_t>(i));
  };

  if (cfg.layers == 1) {
    for (int i = 0; i < n; ++i) {
      topo.adapters.push_back(make_adapter(i));
    }
    topo.switches.push_back(std::make_unique<NetworkSwitch>(
        sim, n, 0, cfg.bw_policy, cfg.lb_policy, cfg.seed));
    NetworkSwitch &sw = *topo.switches[0];
    for (int i = 0; i < n; ++i) {
      link_bidirectional(*topo.adapters[static_cast<std::size_t>(i)], 1, sw, i,
                         cfg.link_bandwidth_bps, cfg.link_delay_s);
    }
  } else {
    int leaf_down = cfg.leaf_downlinks;
    if (leaf_down <= 0) {
      leaf_down = std::max(1, std::min(4, n));
    }
    int num_leaf = cfg.num_leaf;
    if (num_leaf <= 0) {
      num_leaf = (n + leaf_down - 1) / leaf_down;
    }
    if (num_leaf * leaf_down < n) {
      throw std::invalid_argument(
          "fattree 2-layer: num_leaf * leaf_downlinks < endpoints");
    }
    int leaf_up = cfg.leaf_uplinks;
    if (leaf_up <= 0) {
      leaf_up = 1;
    }
    int num_spine = cfg.num_spine;
    if (num_spine <= 0) {
      num_spine = std::max(1, leaf_up);
    }
    if (leaf_up > num_spine) {
      num_spine = leaf_up;
    }

    for (int i = 0; i < n; ++i) {
      topo.adapters.push_back(make_adapter(i));
    }
    const int leaf_ports = leaf_down + leaf_up;
    for (int i = 0; i < num_leaf; ++i) {
      topo.switches.push_back(std::make_unique<NetworkSwitch>(
          sim, leaf_ports, i, cfg.bw_policy, cfg.lb_policy,
          cfg.seed + 1000u + static_cast<uint32_t>(i)));
    }
    for (int i = 0; i < num_spine; ++i) {
      topo.switches.push_back(std::make_unique<NetworkSwitch>(
          sim, num_leaf, num_leaf + i, cfg.bw_policy, cfg.lb_policy,
          cfg.seed + 2000u + static_cast<uint32_t>(i)));
    }

    auto leaf = [&](int i) -> NetworkSwitch & {
      return *topo.switches[static_cast<std::size_t>(i)];
    };
    auto spine = [&](int i) -> NetworkSwitch & {
      return *topo.switches[static_cast<std::size_t>(num_leaf + i)];
    };

    for (int i = 0; i < n; ++i) {
      const int leaf_id = i / leaf_down;
      const int down_port = i % leaf_down;
      link_bidirectional(*topo.adapters[static_cast<std::size_t>(i)], 1,
                         leaf(leaf_id), down_port, cfg.link_bandwidth_bps,
                         cfg.link_delay_s);
    }
    for (int li = 0; li < num_leaf; ++li) {
      for (int u = 0; u < leaf_up; ++u) {
        const int spine_id = u % num_spine;
        const int leaf_up_port = leaf_down + u;
        link_bidirectional(leaf(li), leaf_up_port, spine(spine_id), li,
                           cfg.link_bandwidth_bps, cfg.link_delay_s);
      }
    }
  }

  std::vector<Node *> nodes;
  std::vector<NetworkAdapter *> adapter_ptrs;
  nodes.reserve(topo.adapters.size() + topo.switches.size());
  for (auto &a : topo.adapters) {
    adapter_ptrs.push_back(a.get());
    nodes.push_back(a.get());
  }
  for (auto &s : topo.switches) {
    nodes.push_back(s.get());
  }
  compute_routing_tables(nodes, adapter_ptrs);
  return topo;
}

} // namespace hybridsim::network
