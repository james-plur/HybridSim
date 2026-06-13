#pragma once

#include "hybridsim/engine/kernel_factory.hpp"
#include "hybridsim/engine/workload_spec.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <vector>

namespace hybridsim::engine::detail {

inline simcpp20::process<>
run_node(simcpp20::simulation<> &sim, std::size_t node_id,
         const workload_spec &spec,
         const std::vector<std::vector<std::size_t>> &preds,
         const KernelFactory &factory, std::vector<simcpp20::event<>> &done) {
  for (const auto pred : preds[node_id]) {
    co_await done[pred];
  }

  auto kernel = factory.create(spec.kernels[node_id]);
  co_await kernel->run(sim);
  done[node_id].trigger();
}

} // namespace hybridsim::engine::detail

namespace hybridsim::engine {

inline simcpp20::process<> schedule_dag(simcpp20::simulation<> &sim,
                                        const workload_spec &spec,
                                        const KernelFactory &factory) {
  validate_dag(spec);

  const auto n = spec.kernels.size();
  if (n == 0) {
    co_return;
  }

  const auto preds = build_predecessors(spec);
  std::vector<simcpp20::event<>> done;
  done.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    done.push_back(sim.event());
  }

  std::vector<simcpp20::process<>> node_procs;
  node_procs.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    node_procs.push_back(
        detail::run_node(sim, i, spec, preds, factory, done));
  }

  for (std::size_t i = 0; i < n; ++i) {
    co_await done[i];
  }
}

} // namespace hybridsim::engine
