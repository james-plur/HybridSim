#pragma once

#include "hybridsim/message.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <any>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <utility>
#include <variant>

namespace hybridsim {

/// Base for request envelopes (typed actor + shared reply protocol).
class request_base : public message {
public:
  ~request_base() override = default;

  virtual std::type_index payload_type() const noexcept = 0;
  virtual void *payload_ptr() noexcept = 0;
  virtual bool replied() const noexcept = 0;
  virtual void reply_default() = 0;
  virtual void reply_any(std::any value) = 0;

  std::type_index type() const noexcept override {
    static const std::type_index idx(typeid(request_base));
    return idx;
  }
};

template <typename Msg, typename Reply = std::monostate>
class request_envelope : public request_base {
public:
  Msg payload;
  simcpp20::value_event<Reply> reply_event;

  request_envelope(simcpp20::simulation<> &sim, Msg msg)
      : payload(std::move(msg)), reply_event(sim.template event<Reply>()) {}

  std::type_index payload_type() const noexcept override {
    return typeid(Msg);
  }

  void *payload_ptr() noexcept override { return &payload; }

  bool replied() const noexcept override { return replied_; }

  void reply_default() override {
    if (replied_) {
      return;
    }
    replied_ = true;
    reply_event.trigger(Reply{});
  }

  void reply_any(std::any value) override {
    if (replied_) {
      return;
    }
    replied_ = true;
    reply_event.trigger(std::any_cast<Reply>(std::move(value)));
  }

  void reply_value(Reply value) {
    if (replied_) {
      return;
    }
    replied_ = true;
    reply_event.trigger(std::move(value));
  }

private:
  bool replied_ = false;
};

template <typename Msg>
Msg &message_payload(message &msg) {
  if (auto *typed = dynamic_cast<typed_message<Msg> *>(&msg)) {
    return typed->value;
  }
  if (auto *req = dynamic_cast<request_base *>(&msg)) {
    if (req->payload_type() != typeid(Msg)) {
      throw std::runtime_error("request payload type mismatch");
    }
    return *static_cast<Msg *>(req->payload_ptr());
  }
  throw std::runtime_error("message is not a typed payload or request envelope");
}

} // namespace hybridsim
