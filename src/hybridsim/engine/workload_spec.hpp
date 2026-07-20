#pragma once

#include "hybridsim/engine/kernel_spec.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hybridsim::engine {

struct workload_spec {
  int64_t workload_id = 0;
  std::vector<kernel_spec> kernels;
};

inline std::vector<std::vector<std::size_t>>
build_predecessors(const workload_spec &spec) {
  const auto n = spec.kernels.size();
  std::vector<std::vector<std::size_t>> preds(n);
  for (std::size_t i = 0; i < n; ++i) {
    preds[i] = spec.kernels[i].dependencies;
  }
  return preds;
}

inline void validate_dag(const workload_spec &spec) {
  const auto n = spec.kernels.size();

  for (std::size_t i = 0; i < n; ++i) {
    for (const auto dep : spec.kernels[i].dependencies) {
      if (dep >= n) {
        throw std::invalid_argument("kernel dependency references invalid node id");
      }
      if (dep == i) {
        throw std::invalid_argument("kernel dependency cannot be a self-loop");
      }
    }
  }

  std::vector<std::size_t> in_degree(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    in_degree[i] = spec.kernels[i].dependencies.size();
  }

  std::vector<std::size_t> queue;
  queue.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (in_degree[i] == 0) {
      queue.push_back(i);
    }
  }

  std::size_t visited = 0;
  while (!queue.empty()) {
    const auto node = queue.back();
    queue.pop_back();
    ++visited;

    for (std::size_t i = 0; i < n; ++i) {
      for (const auto dep : spec.kernels[i].dependencies) {
        if (dep != node) {
          continue;
        }
        if (--in_degree[i] == 0) {
          queue.push_back(i);
        }
      }
    }
  }

  if (visited != n) {
    throw std::invalid_argument("dag contains a cycle");
  }
}

} // namespace hybridsim::engine
