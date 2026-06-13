#pragma once

#include "hybridsim/engine/kernel_params.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace hybridsim::engine {

struct kernel_spec {
  std::string name;
  int32_t type = 0;
  double duration = 0.0;
  std::vector<std::size_t> dependencies;
  kernel_params params;
};

} // namespace hybridsim::engine
