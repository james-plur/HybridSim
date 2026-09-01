#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace hybridsim::network {

inline constexpr double kUnlimitedRate = 1e30;
inline constexpr double kDefaultSignalBytes = 64.0;

enum class BwPolicyKind : int32_t {
  MaxMin = 0,
  IngressProportional = 1,
  PriorityThenMaxMin = 2,
};

enum class LbPolicyKind : int32_t {
  EcmpHash = 0,
  Random = 1,
  LeastLoaded = 2,
};

inline bool is_unlimited_rate(double rate) noexcept {
  return !std::isfinite(rate) || rate >= kUnlimitedRate * 0.5;
}

struct NetworkBuildConfig {
  int layers = 1;
  int num_leaf = 0;
  int num_spine = 0;
  int leaf_downlinks = 0;
  int leaf_uplinks = 0;
  double link_bandwidth_bps = 50e9 / 8.0;
  double link_delay_s = 1e-6;
  BwPolicyKind bw_policy = BwPolicyKind::MaxMin;
  LbPolicyKind lb_policy = LbPolicyKind::EcmpHash;
  uint32_t seed = 0;
};

} // namespace hybridsim::network
