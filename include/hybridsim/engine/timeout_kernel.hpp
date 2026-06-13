#pragma once

#include "hybridsim/engine/kernel.hpp"
#include "hybridsim/engine/kernel_spec.hpp"

namespace hybridsim::engine {

class TimeoutKernel : public Kernel {
public:
  explicit TimeoutKernel(kernel_spec spec) : spec_{std::move(spec)} {}

  simcpp20::process<> run(simcpp20::simulation<> &sim) override {
    co_await sim.timeout(spec_.duration);
  }

private:
  kernel_spec spec_;
};

} // namespace hybridsim::engine
