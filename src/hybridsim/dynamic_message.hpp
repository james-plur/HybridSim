#pragma once

#include "hybridsim/message.hpp"

#include "fschuetz04/simcpp20.hpp"

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

/// Shared reply channel for dynamic (Python) request/reply.
struct dynamic_reply_channel {
  explicit dynamic_reply_channel(simcpp20::simulation<> &sim)
      : event(sim.template event<std::any>()) {}

  simcpp20::value_event<std::any> event;
  bool replied = false;

  void complete(std::any value = {}) {
    if (replied) {
      return;
    }
    replied = true;
    event.trigger(std::move(value));
  }
};

class dynamic_message : public message {
public:
  dynamic_message(std::size_t type_id, std::any payload)
      : type_id_{type_id}, payload_{std::move(payload)} {}

  dynamic_message(std::size_t type_id, std::any payload,
                  std::shared_ptr<dynamic_reply_channel> reply)
      : type_id_{type_id}, payload_{std::move(payload)},
        reply_{std::move(reply)} {}

  std::type_index type() const noexcept override {
    static const std::type_index idx(typeid(dynamic_message));
    return idx;
  }

  std::size_t type_id() const noexcept { return type_id_; }

  std::any &payload() noexcept { return payload_; }
  const std::any &payload() const noexcept { return payload_; }

  bool is_request() const noexcept { return static_cast<bool>(reply_); }

  dynamic_reply_channel *reply_channel() const noexcept {
    return reply_.get();
  }

  const std::shared_ptr<dynamic_reply_channel> &reply_channel_ptr() const noexcept {
    return reply_;
  }

private:
  std::size_t type_id_;
  std::any payload_;
  std::shared_ptr<dynamic_reply_channel> reply_;
};

inline std::shared_ptr<dynamic_message>
make_dynamic_message(std::size_t type_id, std::any payload) {
  return std::make_shared<dynamic_message>(type_id, std::move(payload));
}

inline std::shared_ptr<dynamic_message>
make_dynamic_request(std::size_t type_id, std::any payload,
                     simcpp20::simulation<> &sim) {
  auto channel = std::make_shared<dynamic_reply_channel>(sim);
  return std::make_shared<dynamic_message>(type_id, std::move(payload),
                                           std::move(channel));
}

} // namespace hybridsim
