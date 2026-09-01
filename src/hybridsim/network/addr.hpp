#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace hybridsim::network {

struct NetworkAddr {
  int32_t replica_id = 0;
  int32_t rank = 0;

  static NetworkAddr parse(const std::string &s) {
    NetworkAddr addr;
    const auto pos = s.find(':');
    if (pos == std::string::npos) {
      addr.replica_id = 0;
      addr.rank = static_cast<int32_t>(std::stoi(s));
      return addr;
    }
    addr.replica_id = static_cast<int32_t>(std::stoi(s.substr(0, pos)));
    addr.rank = static_cast<int32_t>(std::stoi(s.substr(pos + 1)));
    return addr;
  }

  std::string to_string() const {
    return std::to_string(replica_id) + ":" + std::to_string(rank);
  }

  uint64_t key() const noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(replica_id)) << 32) |
           static_cast<uint32_t>(rank);
  }
};

inline bool operator==(const NetworkAddr &a, const NetworkAddr &b) noexcept {
  return a.replica_id == b.replica_id && a.rank == b.rank;
}

inline bool operator!=(const NetworkAddr &a, const NetworkAddr &b) noexcept {
  return !(a == b);
}

} // namespace hybridsim::network

namespace std {

template <>
struct hash<hybridsim::network::NetworkAddr> {
  size_t operator()(const hybridsim::network::NetworkAddr &addr) const noexcept {
    return std::hash<uint64_t>{}(addr.key());
  }
};

} // namespace std
