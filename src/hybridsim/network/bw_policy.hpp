#pragma once

#include "hybridsim/network/flow.hpp"
#include "hybridsim/network/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hybridsim::network {

namespace detail {

inline bool rates_changed(const std::vector<double> &old_rates,
                          const std::vector<double> &new_rates) {
  if (old_rates.size() != new_rates.size()) {
    return true;
  }
  for (std::size_t i = 0; i < old_rates.size(); ++i) {
    if (std::abs(old_rates[i] - new_rates[i]) > 1e-12) {
      return true;
    }
  }
  return false;
}

inline std::vector<FlowInfo *> flow_ptrs(
    std::unordered_map<uint64_t, FlowInfo> &flows) {
  std::vector<FlowInfo *> out;
  out.reserve(flows.size());
  for (auto &[id, info] : flows) {
    (void)id;
    out.push_back(&info);
  }
  return out;
}

inline void snapshot_rates(const std::vector<FlowInfo *> &ptrs,
                           std::vector<double> &out) {
  out.clear();
  out.reserve(ptrs.size());
  for (auto *f : ptrs) {
    out.push_back(f->rate_bps);
  }
}

/// Max-min fair share of ``capacity``, each flow capped by ``ingress_rate``.
inline void max_min_allocate(std::vector<FlowInfo *> &flows, double capacity) {
  const int n = static_cast<int>(flows.size());
  if (n == 0) {
    return;
  }
  std::vector<char> done(static_cast<std::size_t>(n), 0);
  std::vector<double> want(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    want[static_cast<std::size_t>(i)] =
        is_unlimited_rate(flows[static_cast<std::size_t>(i)]->ingress_rate)
            ? kUnlimitedRate
            : std::max(0.0, flows[static_cast<std::size_t>(i)]->ingress_rate);
    flows[static_cast<std::size_t>(i)]->rate_bps = 0.0;
  }

  double remaining = std::max(0.0, capacity);
  int left = n;
  while (left > 0 && remaining > 0.0) {
    const double fair = remaining / static_cast<double>(left);
    bool progressed = false;
    for (int i = 0; i < n; ++i) {
      if (done[static_cast<std::size_t>(i)]) {
        continue;
      }
      const double cap_i = want[static_cast<std::size_t>(i)];
      if (cap_i + 1e-18 < fair) {
        flows[static_cast<std::size_t>(i)]->rate_bps = cap_i;
        remaining -= cap_i;
        done[static_cast<std::size_t>(i)] = 1;
        --left;
        progressed = true;
      }
    }
    if (progressed) {
      continue;
    }
    for (int i = 0; i < n; ++i) {
      if (done[static_cast<std::size_t>(i)]) {
        continue;
      }
      const double cap_i = want[static_cast<std::size_t>(i)];
      flows[static_cast<std::size_t>(i)]->rate_bps = std::min(fair, cap_i);
      remaining -= flows[static_cast<std::size_t>(i)]->rate_bps;
      done[static_cast<std::size_t>(i)] = 1;
      --left;
    }
  }
}

inline void ingress_proportional_allocate(std::vector<FlowInfo *> &flows,
                                          double capacity) {
  double sum = 0.0;
  for (auto *f : flows) {
    const double w =
        is_unlimited_rate(f->ingress_rate) ? 1.0 : std::max(0.0, f->ingress_rate);
    sum += w;
  }
  if (sum <= 0.0) {
    const double fair =
        flows.empty() ? 0.0 : capacity / static_cast<double>(flows.size());
    for (auto *f : flows) {
      f->rate_bps = fair;
    }
    return;
  }
  if (sum <= capacity) {
    for (auto *f : flows) {
      f->rate_bps = is_unlimited_rate(f->ingress_rate)
                        ? (capacity / static_cast<double>(flows.size()))
                        : f->ingress_rate;
    }
    // Unlimited flows share leftover after limited ones take their ingress.
    double used = 0.0;
    int unlimited = 0;
    for (auto *f : flows) {
      if (is_unlimited_rate(f->ingress_rate)) {
        ++unlimited;
      } else {
        used += f->rate_bps;
      }
    }
    if (unlimited > 0) {
      const double leftover = std::max(0.0, capacity - used);
      const double each = leftover / static_cast<double>(unlimited);
      for (auto *f : flows) {
        if (is_unlimited_rate(f->ingress_rate)) {
          f->rate_bps = each;
        }
      }
    }
    return;
  }
  const double scale = capacity / sum;
  for (auto *f : flows) {
    const double w =
        is_unlimited_rate(f->ingress_rate) ? 1.0 : std::max(0.0, f->ingress_rate);
    f->rate_bps = w * scale;
  }
}

inline void priority_then_maxmin(std::vector<FlowInfo *> &flows,
                                 double capacity) {
  if (flows.empty()) {
    return;
  }
  int min_q = flows[0]->qos;
  int max_q = flows[0]->qos;
  for (auto *f : flows) {
    min_q = std::min(min_q, f->qos);
    max_q = std::max(max_q, f->qos);
  }
  double remaining = std::max(0.0, capacity);
  for (int q = max_q; q >= min_q; --q) {
    std::vector<FlowInfo *> cls;
    for (auto *f : flows) {
      if (f->qos == q) {
        cls.push_back(f);
      }
    }
    if (cls.empty()) {
      continue;
    }
    max_min_allocate(cls, remaining);
    for (auto *f : cls) {
      remaining -= f->rate_bps;
    }
    remaining = std::max(0.0, remaining);
  }
}

} // namespace detail

/// Advance remaining bytes, recompute rates. Returns true if any rate changed.
inline bool allocate_bw(BwPolicyKind kind,
                        std::unordered_map<uint64_t, FlowInfo> &flows,
                        double capacity, double now) {
  auto ptrs = detail::flow_ptrs(flows);
  std::vector<double> old_rates;
  detail::snapshot_rates(ptrs, old_rates);
  for (auto *f : ptrs) {
    f->advance(now);
  }
  switch (kind) {
  case BwPolicyKind::IngressProportional:
    detail::ingress_proportional_allocate(ptrs, capacity);
    break;
  case BwPolicyKind::PriorityThenMaxMin:
    detail::priority_then_maxmin(ptrs, capacity);
    break;
  case BwPolicyKind::MaxMin:
  default:
    detail::max_min_allocate(ptrs, capacity);
    break;
  }
  std::vector<double> new_rates;
  detail::snapshot_rates(ptrs, new_rates);
  return detail::rates_changed(old_rates, new_rates);
}

} // namespace hybridsim::network
