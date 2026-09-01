#pragma once

#include "hybridsim/network/addr.hpp"
#include "hybridsim/network/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace hybridsim::network {

class OutPort;

struct FlowInfo {
  uint64_t flow_id = 0;
  uint64_t version = 0;
  NetworkAddr src;
  NetworkAddr dst;
  int64_t conn_id = 0;
  int qos = 0;
  double size_bytes = 0.0;
  double remaining_bytes = 0.0;
  double rate_bps = 0.0;
  double ingress_rate = kUnlimitedRate;
  double last_update = 0.0;
  bool is_fetch = false;
  double fetch_payload_bytes = 0.0;
  OutPort *out_port = nullptr;

  void advance(double now) {
    if (now <= last_update) {
      last_update = std::max(last_update, now);
      return;
    }
    if (rate_bps > 0.0 && !is_unlimited_rate(rate_bps)) {
      remaining_bytes -= rate_bps * (now - last_update);
      remaining_bytes = std::max(0.0, remaining_bytes);
    } else if (is_unlimited_rate(rate_bps)) {
      remaining_bytes = 0.0;
    }
    last_update = now;
  }
};

struct FlowArriveMsg {
  uint64_t flow_id = 0;
  NetworkAddr src;
  NetworkAddr dst;
  int64_t conn_id = 0;
  int qos = 0;
  double size_bytes = 0.0;
  double rate_bps = kUnlimitedRate;
  bool is_fetch = false;
  double fetch_payload_bytes = 0.0;
};

struct FlowUpdateMsg {
  uint64_t flow_id = 0;
  NetworkAddr src;
  NetworkAddr dst;
  int64_t conn_id = 0;
  int qos = 0;
  double remaining_bytes = 0.0;
  double rate_bps = 0.0;
  bool is_fetch = false;
  double fetch_payload_bytes = 0.0;
};

struct FlowEndMsg {
  uint64_t flow_id = 0;
  uint64_t version = 0;
};

inline FlowInfo flow_from_arrive(const FlowArriveMsg &msg, double now) {
  FlowInfo info;
  info.flow_id = msg.flow_id;
  info.src = msg.src;
  info.dst = msg.dst;
  info.conn_id = msg.conn_id;
  info.qos = msg.qos;
  info.size_bytes = msg.size_bytes;
  info.remaining_bytes = msg.size_bytes;
  info.rate_bps = 0.0;
  info.ingress_rate =
      is_unlimited_rate(msg.rate_bps) ? kUnlimitedRate : msg.rate_bps;
  info.last_update = now;
  info.is_fetch = msg.is_fetch;
  info.fetch_payload_bytes = msg.fetch_payload_bytes;
  return info;
}

inline FlowArriveMsg arrive_from_flow(const FlowInfo &info) {
  FlowArriveMsg msg;
  msg.flow_id = info.flow_id;
  msg.src = info.src;
  msg.dst = info.dst;
  msg.conn_id = info.conn_id;
  msg.qos = info.qos;
  msg.size_bytes = info.size_bytes;
  msg.rate_bps =
      is_unlimited_rate(info.rate_bps) ? kUnlimitedRate : info.rate_bps;
  msg.is_fetch = info.is_fetch;
  msg.fetch_payload_bytes = info.fetch_payload_bytes;
  return msg;
}

inline FlowUpdateMsg update_from_flow(const FlowInfo &info) {
  FlowUpdateMsg msg;
  msg.flow_id = info.flow_id;
  msg.src = info.src;
  msg.dst = info.dst;
  msg.conn_id = info.conn_id;
  msg.qos = info.qos;
  msg.remaining_bytes = info.remaining_bytes;
  msg.rate_bps =
      is_unlimited_rate(info.rate_bps) ? kUnlimitedRate : info.rate_bps;
  msg.is_fetch = info.is_fetch;
  msg.fetch_payload_bytes = info.fetch_payload_bytes;
  return msg;
}

inline double transmission_dt(double remaining_bytes, double rate_bps) {
  if (remaining_bytes <= 0.0) {
    return 0.0;
  }
  if (rate_bps <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (is_unlimited_rate(rate_bps)) {
    return 0.0;
  }
  return remaining_bytes / rate_bps;
}

} // namespace hybridsim::network
