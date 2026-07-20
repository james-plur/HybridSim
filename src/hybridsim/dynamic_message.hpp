#pragma once

#include "hybridsim/message.hpp"

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hybridsim {

class message_registry {
public:
  static message_registry &instance() {
    static message_registry reg;
    return reg;
  }

  std::size_t register_type(const std::string &name) {
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
      return it->second;
    }
    const std::size_t id = next_id_++;
    name_to_id_[name] = id;
    id_to_name_[id] = name;
    return id;
  }

  std::size_t lookup(const std::string &name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
      throw std::runtime_error("unknown message type: " + name);
    }
    return it->second;
  }

  const std::string &name(std::size_t id) const {
    auto it = id_to_name_.find(id);
    if (it == id_to_name_.end()) {
      throw std::runtime_error("unknown message type id");
    }
    return it->second;
  }

private:
  std::size_t next_id_ = 1;
  std::unordered_map<std::string, std::size_t> name_to_id_;
  std::unordered_map<std::size_t, std::string> id_to_name_;
};

class dynamic_message : public message {
public:
  dynamic_message(std::size_t type_id, std::any payload)
      : type_id_{type_id}, payload_{std::move(payload)} {}

  std::type_index type() const noexcept override {
    static const std::type_index idx(typeid(dynamic_message));
    return idx;
  }

  std::size_t type_id() const noexcept { return type_id_; }

  std::any &payload() noexcept { return payload_; }
  const std::any &payload() const noexcept { return payload_; }

private:
  std::size_t type_id_;
  std::any payload_;
};

inline std::shared_ptr<dynamic_message>
make_dynamic_message(std::size_t type_id, std::any payload) {
  return std::make_shared<dynamic_message>(type_id, std::move(payload));
}

} // namespace hybridsim
