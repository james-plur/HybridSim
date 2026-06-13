#pragma once

#include "hybridsim/engine/workload_spec.hpp"

namespace hybridsim::engine {

struct WorkloadMsg {
  workload_spec spec;
};

struct WorkloadDoneMsg {
  int64_t workload_id = 0;
};

} // namespace hybridsim::engine
