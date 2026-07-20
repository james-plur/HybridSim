#pragma once

#include "fschuetz04/simcpp20.hpp"

namespace hybridsim::engine {

class Kernel {
public:
  virtual ~Kernel() = default;
  virtual simcpp20::process<> run(simcpp20::simulation<> &sim) = 0;
};

} // namespace hybridsim::engine
