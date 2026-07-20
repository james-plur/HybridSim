#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace hybridsim::engine {

using kernel_param_value = std::variant<bool, int64_t, double, std::string>;

class kernel_params {
public:
  bool empty() const noexcept { return values_.empty(); }

  bool contains(const std::string &key) const {
    return values_.find(key) != values_.end();
  }

  void clear() noexcept { values_.clear(); }

  void set_bool(const std::string &key, bool value) { values_[key] = value; }

  void set_int(const std::string &key, int64_t value) { values_[key] = value; }

  void set_double(const std::string &key, double value) { values_[key] = value; }

  void set_string(const std::string &key, std::string value) {
    values_[key] = std::move(value);
  }

  std::optional<bool> get_bool(const std::string &key) const {
    return get_as<bool>(key);
  }

  std::optional<int64_t> get_int(const std::string &key) const {
    return get_as<int64_t>(key);
  }

  std::optional<double> get_double(const std::string &key) const {
    return get_as<double>(key);
  }

  std::optional<std::string> get_string(const std::string &key) const {
    return get_as<std::string>(key);
  }

  const std::unordered_map<std::string, kernel_param_value> &values() const noexcept {
    return values_;
  }

  std::unordered_map<std::string, kernel_param_value> &values() noexcept {
    return values_;
  }

private:
  template <typename T>
  std::optional<T> get_as(const std::string &key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      return std::nullopt;
    }
    if (const auto *value = std::get_if<T>(&it->second)) {
      return *value;
    }
    return std::nullopt;
  }

  std::unordered_map<std::string, kernel_param_value> values_;
};

} // namespace hybridsim::engine
