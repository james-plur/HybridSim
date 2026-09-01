#pragma once

#include "hybridsim/network/node.hpp"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace hybridsim::network {

class NetworkAdapter : public Node {
public:
  NetworkAdapter(simcpp20::simulation<> &sim, NetworkAddr addr, int port_num,
                 int index, BwPolicyKind bw, LbPolicyKind lb, uint32_t seed)
      : Node(sim, port_num, bw, lb, seed), addr_(addr), index_(index) {}

  bool is_endpoint() const override { return true; }
  NetworkAddr addr() const override { return addr_; }
  int index() const noexcept { return index_; }

  InPort &host_in() { return in_port(0); }
  OutPort &host_out() { return out_port(0); }

  uint64_t new_flow_id() {
    return (static_cast<uint64_t>(index_ + 1) << 32) | (++flow_counter_);
  }

  void inject(const NetworkAddr &dst, int64_t conn_id, int qos,
              double size_bytes, bool is_fetch = false,
              double fetch_payload_bytes = 0.0) {
    FlowArriveMsg msg;
    msg.flow_id = new_flow_id();
    msg.src = addr_;
    msg.dst = dst;
    msg.conn_id = conn_id;
    msg.qos = qos;
    msg.size_bytes = size_bytes;
    msg.rate_bps = kUnlimitedRate;
    msg.is_fetch = is_fetch;
    msg.fetch_payload_bytes = fetch_payload_bytes;
    host_in().send(msg, 0.0);
  }

  simcpp20::event<> recv_event(int64_t conn_id) {
    auto &pend = pending_[conn_id];
    if (unmatched_[conn_id] > 0) {
      --unmatched_[conn_id];
      auto ev = sim().event();
      ev.trigger();
      return ev;
    }
    auto ev = sim().event();
    pend.push(ev);
    return ev;
  }

  void on_local_flow_complete(const FlowInfo &info) override {
    if (info.is_fetch) {
      inject(info.src, info.conn_id, info.qos,
             std::max(0.0, info.fetch_payload_bytes), /*is_fetch=*/false);
      return;
    }
    complete_recv(info.conn_id);
  }

private:
  void complete_recv(int64_t conn_id) {
    auto it = pending_.find(conn_id);
    if (it != pending_.end() && !it->second.empty()) {
      auto ev = it->second.front();
      it->second.pop();
      ev.trigger();
      return;
    }
    ++unmatched_[conn_id];
  }

  NetworkAddr addr_{};
  int index_ = 0;
  uint64_t flow_counter_ = 0;
  std::unordered_map<int64_t, std::queue<simcpp20::event<>>> pending_;
  std::unordered_map<int64_t, int> unmatched_;
};

class NetworkSwitch : public Node {
public:
  NetworkSwitch(simcpp20::simulation<> &sim, int port_num, int index,
                BwPolicyKind bw, LbPolicyKind lb, uint32_t seed)
      : Node(sim, port_num, bw, lb, seed), index_(index) {}

  int index() const noexcept { return index_; }

private:
  int index_ = 0;
};

} // namespace hybridsim::network
