#pragma once

#include <cstdint>

namespace hybridsim::engine {

inline constexpr int32_t kKernelTimeout = 0;
inline constexpr int32_t kKernelPut = 1;
inline constexpr int32_t kKernelSignal = 2;
inline constexpr int32_t kKernelWait = 3;
inline constexpr int32_t kKernelGet = 4;

} // namespace hybridsim::engine
