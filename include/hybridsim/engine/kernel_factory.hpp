#pragma once

#include "hybridsim/engine/kernel.hpp"
#include "hybridsim/engine/kernel_spec.hpp"
#include "hybridsim/engine/timeout_kernel.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace hybridsim::engine {

class KernelFactory {
public:
  using CreatorFn = std::function<std::unique_ptr<Kernel>(const kernel_spec &)>;

  KernelFactory() { register_default_creators(); }

  void register_creator(int32_t type, CreatorFn creator) {
    creators_[type] = std::move(creator);
  }

  std::unique_ptr<Kernel> create(const kernel_spec &spec) const {
    const auto it = creators_.find(spec.type);
    if (it == creators_.end()) {
      throw std::invalid_argument("unknown kernel type: " +
                                  std::to_string(spec.type));
    }
    return it->second(spec);
  }

private:
  void register_default_creators() {
    register_creator(0, [](const kernel_spec &spec) {
      return std::make_unique<TimeoutKernel>(spec);
    });
  }

  std::unordered_map<int32_t, CreatorFn> creators_;
};

} // namespace hybridsim::engine
