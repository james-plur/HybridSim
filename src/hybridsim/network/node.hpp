#pragma once

#include "hybridsim/actor.hpp"
#include "hybridsim/network/addr.hpp"
#include "hybridsim/network/bw_policy.hpp"
#include "hybridsim/network/flow.hpp"
#include "hybridsim/network/types.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hybridsim::network {

class Node;
class OutPort;
class InPort;

struct ForwardResult {
  OutPort *out = nullptr;
};

class InPort : public actor {
public:
  InPort(simcpp20::simulation<> &sim, Node &node, int index);

  int index() const noexcept { return index_; }
  Node &owner() noexcept { return *node_; }

  std::unordered_map<uint64_t, FlowInfo> &flows() noexcept { return flows_; }
  const std::unordered_map<uint64_t, FlowInfo> &flows() const noexcept {
    return flows_;
  }

private:
  void on_arrive(FlowArriveMsg &msg);
  void on_update(FlowUpdateMsg &msg);
  void on_end(FlowEndMsg &msg);
  void schedule_end(FlowInfo &info);

  Node *node_ = nullptr;
  int index_ = 0;
  std::unordered_map<uint64_t, FlowInfo> flows_;
};

class OutPort : public actor {
public:
  OutPort(simcpp20::simulation<> &sim, Node &node, int index);

  int index() const noexcept { return index_; }
  Node &owner() noexcept { return *node_; }

  void set_downstream(InPort *peer, double bandwidth_bps, double delay_s) {
    downstream_ = peer;
    bandwidth_bps_ = bandwidth_bps;
    delay_s_ = delay_s;
  }

  InPort *downstream() const noexcept { return downstream_; }
  double bandwidth_bps() const noexcept { return bandwidth_bps_; }
  double delay_s() const noexcept { return delay_s_; }
  std::size_t flow_count() const noexcept { return flows_.size(); }

  double used_rate() const {
    double sum = 0.0;
    for (const auto &[id, info] : flows_) {
      (void)id;
      if (!is_unlimited_rate(info.rate_bps)) {
        sum += info.rate_bps;
      }
    }
    return sum;
  }

  std::unordered_map<uint64_t, FlowInfo> &flows() noexcept { return flows_; }
  const std::unordered_map<uint64_t, FlowInfo> &flows() const noexcept {
    return flows_;
  }

  void deliver_internal(const FlowArriveMsg &msg) { send(msg, 0.0); }
  void deliver_internal(const FlowUpdateMsg &msg) { send(msg, 0.0); }

private:
  void on_arrive(FlowArriveMsg &msg);
  void on_update(FlowUpdateMsg &msg);
  void on_end(FlowEndMsg &msg);
  void schedule_end(FlowInfo &info);
  void send_update(const FlowInfo &info);
  std::unordered_map<uint64_t, double> snapshot_rates() const;

  Node *node_ = nullptr;
  int index_ = 0;
  InPort *downstream_ = nullptr;
  double bandwidth_bps_ = kUnlimitedRate;
  double delay_s_ = 0.0;
  std::unordered_map<uint64_t, FlowInfo> flows_;
};

class Router {
public:
  explicit Router(LbPolicyKind lb = LbPolicyKind::EcmpHash, uint32_t seed = 0)
      : lb_(lb), rng_(seed) {}

  void set_nexthops(const NetworkAddr &dst, std::vector<int> ports) {
    table_[dst.key()] = std::move(ports);
  }

  const std::vector<int> *nexthops(const NetworkAddr &dst) const {
    const auto it = table_.find(dst.key());
    if (it == table_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  OutPort *route(Node &node, const FlowArriveMsg &msg);

  LbPolicyKind lb_policy() const noexcept { return lb_; }

private:
  LbPolicyKind lb_ = LbPolicyKind::EcmpHash;
  std::mt19937 rng_;
  std::unordered_map<uint64_t, std::vector<int>> table_;
};

class Node {
public:
  Node(simcpp20::simulation<> &sim, int port_num, BwPolicyKind bw,
       LbPolicyKind lb, uint32_t seed)
      : sim_(&sim), bw_policy_(bw), router_(lb, seed) {
    in_ports_.reserve(static_cast<std::size_t>(port_num));
    out_ports_.reserve(static_cast<std::size_t>(port_num));
    for (int i = 0; i < port_num; ++i) {
      in_ports_.push_back(std::make_unique<InPort>(sim, *this, i));
      out_ports_.push_back(std::make_unique<OutPort>(sim, *this, i));
    }
  }

  virtual ~Node() = default;

  simcpp20::simulation<> &sim() noexcept { return *sim_; }

  int port_num() const noexcept { return static_cast<int>(in_ports_.size()); }

  InPort &in_port(int i) { return *in_ports_.at(static_cast<std::size_t>(i)); }
  OutPort &out_port(int i) {
    return *out_ports_.at(static_cast<std::size_t>(i));
  }

  Router &router() noexcept { return router_; }
  const Router &router() const noexcept { return router_; }

  virtual bool is_endpoint() const { return false; }
  virtual NetworkAddr addr() const { return {}; }
  virtual void on_local_flow_complete(const FlowInfo &) {}

  virtual ForwardResult forward(const FlowArriveMsg &msg) {
    if (is_endpoint() && addr() == msg.dst) {
      return ForwardResult{};
    }
    return ForwardResult{router_.route(*this, msg)};
  }

  virtual bool allocate_bw(OutPort &port, double now) {
    return hybridsim::network::allocate_bw(bw_policy_, port.flows(),
                                           port.bandwidth_bps(), now);
  }

  void start_ports() {
    for (auto &p : in_ports_) {
      p->start();
    }
    for (auto &p : out_ports_) {
      p->start();
    }
  }

  void stop_ports() {
    for (auto &p : in_ports_) {
      p->stop();
    }
    for (auto &p : out_ports_) {
      p->stop();
    }
  }

  void rethrow_port_errors() const {
    for (const auto &p : in_ports_) {
      p->rethrow_if_error();
    }
    for (const auto &p : out_ports_) {
      p->rethrow_if_error();
    }
  }

protected:
  simcpp20::simulation<> *sim_ = nullptr;
  BwPolicyKind bw_policy_ = BwPolicyKind::MaxMin;
  Router router_;
  std::vector<std::unique_ptr<InPort>> in_ports_;
  std::vector<std::unique_ptr<OutPort>> out_ports_;
};

inline InPort::InPort(simcpp20::simulation<> &sim, Node &node, int index)
    : actor(sim), node_(&node), index_(index) {
  on<FlowArriveMsg>([this](actor &, FlowArriveMsg &msg) { on_arrive(msg); });
  on<FlowUpdateMsg>([this](actor &, FlowUpdateMsg &msg) { on_update(msg); });
  on<FlowEndMsg>([this](actor &, FlowEndMsg &msg) { on_end(msg); });
}

inline void InPort::schedule_end(FlowInfo &info) {
  ++info.version;
  const double dt = transmission_dt(info.remaining_bytes, info.rate_bps);
  if (!std::isfinite(dt)) {
    return;
  }
  send_at(sim().now() + dt, FlowEndMsg{info.flow_id, info.version});
}

inline void InPort::on_arrive(FlowArriveMsg &msg) {
  const ForwardResult fwd = node_->forward(msg);
  FlowInfo info = flow_from_arrive(msg, sim().now());
  info.rate_bps = is_unlimited_rate(msg.rate_bps) ? kUnlimitedRate : msg.rate_bps;
  info.out_port = fwd.out;
  flows_[info.flow_id] = info;

  if (fwd.out != nullptr) {
    fwd.out->deliver_internal(msg);
    schedule_end(flows_[info.flow_id]);
    return;
  }
  if (node_->is_endpoint() && node_->addr() == msg.dst) {
    schedule_end(flows_[info.flow_id]);
    return;
  }
  flows_.erase(info.flow_id);
}

inline void InPort::on_update(FlowUpdateMsg &msg) {
  const auto it = flows_.find(msg.flow_id);
  if (it == flows_.end()) {
    return;
  }
  it->second.advance(sim().now());
  it->second.rate_bps =
      is_unlimited_rate(msg.rate_bps) ? kUnlimitedRate : msg.rate_bps;
  if (it->second.out_port != nullptr) {
    it->second.out_port->deliver_internal(msg);
    schedule_end(it->second);
    return;
  }
  schedule_end(it->second);
}

inline void InPort::on_end(FlowEndMsg &msg) {
  const auto it = flows_.find(msg.flow_id);
  if (it == flows_.end()) {
    return;
  }
  if (it->second.version != msg.version) {
    return;
  }
  FlowInfo info = it->second;
  flows_.erase(it);
  if (info.out_port == nullptr && node_->is_endpoint() &&
      node_->addr() == info.dst) {
    node_->on_local_flow_complete(info);
  }
}

inline OutPort::OutPort(simcpp20::simulation<> &sim, Node &node, int index)
    : actor(sim), node_(&node), index_(index) {
  on<FlowArriveMsg>([this](actor &, FlowArriveMsg &msg) { on_arrive(msg); });
  on<FlowUpdateMsg>([this](actor &, FlowUpdateMsg &msg) { on_update(msg); });
  on<FlowEndMsg>([this](actor &, FlowEndMsg &msg) { on_end(msg); });
}

inline void OutPort::schedule_end(FlowInfo &info) {
  ++info.version;
  const double dt = transmission_dt(info.remaining_bytes, info.rate_bps);
  if (!std::isfinite(dt)) {
    return;
  }
  send_at(sim().now() + dt, FlowEndMsg{info.flow_id, info.version});
}

inline void OutPort::send_update(const FlowInfo &info) {
  if (downstream_ == nullptr) {
    return;
  }
  downstream_->send(update_from_flow(info), delay_s_);
}

inline std::unordered_map<uint64_t, double> OutPort::snapshot_rates() const {
  std::unordered_map<uint64_t, double> out;
  out.reserve(flows_.size());
  for (const auto &[id, info] : flows_) {
    out[id] = info.rate_bps;
  }
  return out;
}

inline void OutPort::on_arrive(FlowArriveMsg &msg) {
  FlowInfo info = flow_from_arrive(msg, sim().now());
  const auto old = snapshot_rates();
  flows_[info.flow_id] = info;
  node_->allocate_bw(*this, sim().now());

  for (auto &[id, flow] : flows_) {
    if (id == info.flow_id) {
      continue;
    }
    const auto it = old.find(id);
    if (it == old.end() || std::abs(it->second - flow.rate_bps) > 1e-12) {
      send_update(flow);
      schedule_end(flow);
    }
  }
  schedule_end(flows_[info.flow_id]);
  if (downstream_ != nullptr) {
    downstream_->send(arrive_from_flow(flows_[info.flow_id]), delay_s_);
  }
}

inline void OutPort::on_update(FlowUpdateMsg &msg) {
  const auto it = flows_.find(msg.flow_id);
  if (it == flows_.end()) {
    return;
  }
  it->second.advance(sim().now());
  it->second.ingress_rate =
      is_unlimited_rate(msg.rate_bps) ? kUnlimitedRate : msg.rate_bps;
  const auto old = snapshot_rates();
  node_->allocate_bw(*this, sim().now());
  for (auto &[id, flow] : flows_) {
    const auto oit = old.find(id);
    if (oit == old.end() || std::abs(oit->second - flow.rate_bps) > 1e-12) {
      send_update(flow);
      schedule_end(flow);
    }
  }
}

inline void OutPort::on_end(FlowEndMsg &msg) {
  const auto it = flows_.find(msg.flow_id);
  if (it == flows_.end()) {
    return;
  }
  if (it->second.version != msg.version) {
    return;
  }
  flows_.erase(it);
  const auto old = snapshot_rates();
  node_->allocate_bw(*this, sim().now());
  for (auto &[id, flow] : flows_) {
    const auto oit = old.find(id);
    if (oit == old.end() || std::abs(oit->second - flow.rate_bps) > 1e-12) {
      send_update(flow);
      schedule_end(flow);
    }
  }
}

inline OutPort *Router::route(Node &node, const FlowArriveMsg &msg) {
  const auto *hops = nexthops(msg.dst);
  if (hops == nullptr || hops->empty()) {
    return nullptr;
  }
  int idx = 0;
  switch (lb_) {
  case LbPolicyKind::Random: {
    std::uniform_int_distribution<int> dist(
        0, static_cast<int>(hops->size()) - 1);
    idx = dist(rng_);
    break;
  }
  case LbPolicyKind::LeastLoaded: {
    int best = (*hops)[0];
    double best_load = node.out_port(best).used_rate();
    std::size_t best_count = node.out_port(best).flow_count();
    for (std::size_t i = 1; i < hops->size(); ++i) {
      const int p = (*hops)[i];
      const double load = node.out_port(p).used_rate();
      const std::size_t count = node.out_port(p).flow_count();
      if (load < best_load - 1e-12 ||
          (std::abs(load - best_load) <= 1e-12 && count < best_count)) {
        best = p;
        best_load = load;
        best_count = count;
      }
    }
    idx = -1;
    for (std::size_t i = 0; i < hops->size(); ++i) {
      if ((*hops)[i] == best) {
        return &node.out_port(best);
      }
    }
    (void)idx;
    return &node.out_port(best);
  }
  case LbPolicyKind::EcmpHash:
  default:
    idx = static_cast<int>(std::hash<uint64_t>{}(msg.flow_id) % hops->size());
    break;
  }
  return &node.out_port((*hops)[static_cast<std::size_t>(idx)]);
}

} // namespace hybridsim::network
