#pragma once

#include "hybridsim/dynamic_message.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace hybridsim {

class dynamic_actor;

using dynamic_handler_dispatcher = std::function<simcpp20::process<>(
    simcpp20::simulation<> &, dynamic_actor &,
    std::shared_ptr<dynamic_message>)>;

class dynamic_actor {
public:
  explicit dynamic_actor(simcpp20::simulation<> &sim)
      : sim_{sim}, mailbox_{sim}, run_process_{sim} {}

  void on(std::size_t type_id, dynamic_handler_dispatcher dispatcher) {
    handlers_[type_id] = std::move(dispatcher);
  }

  void send(std::shared_ptr<dynamic_message> msg, double delay = 0.0) {
    if (delay > 0.0) {
      send_at(sim_.now() + delay, std::move(msg));
      return;
    }
    mailbox_.put(std::move(msg));
  }

  // Deliver at simulation time `when` (immediate if when <= now).
  void send_at(double when, std::shared_ptr<dynamic_message> msg) {
    if (when <= sim_.now()) {
      send(std::move(msg));
      return;
    }
    delayed_deliver(sim_, *this, when, std::move(msg));
  }

  /// Request/reply: wraps payload with a reply channel; await the returned event.
  /// If ``delay`` > 0, the receiver sees the request after ``delay`` sim time.
  simcpp20::value_event<std::any>
  request(std::size_t type_id, std::any payload, double delay = 0.0) {
    auto msg = make_dynamic_request(type_id, std::move(payload), sim_);
    auto event = msg->reply_channel()->event;
    send(std::move(msg), delay);
    return event;
  }

  simcpp20::value_event<std::any>
  request_at(double when, std::size_t type_id, std::any payload) {
    auto msg = make_dynamic_request(type_id, std::move(payload), sim_);
    auto event = msg->reply_channel()->event;
    send_at(when, std::move(msg));
    return event;
  }

  dynamic_message *current_request() const noexcept {
    return inflight_request_;
  }

  void reply(std::any value = {}) {
    if (!inflight_request_ || !inflight_request_->is_request()) {
      throw std::runtime_error("reply() called without an in-flight request");
    }
    inflight_request_->reply_channel()->complete(std::move(value));
  }

  void reply(dynamic_message &req, std::any value = {}) {
    if (!req.is_request()) {
      throw std::runtime_error("reply() on a non-request message");
    }
    req.reply_channel()->complete(std::move(value));
  }

  void reply_at(double when, std::any value = {}) {
    if (!inflight_message_ || !inflight_message_->is_request()) {
      throw std::runtime_error("reply_at() without an in-flight request");
    }
    if (when <= sim_.now()) {
      reply(std::move(value));
      return;
    }
    delayed_reply(sim_, inflight_message_, when, std::move(value));
  }

  void reply_at(double when, dynamic_message &req, std::any value = {}) {
    if (!req.is_request()) {
      throw std::runtime_error("reply_at() on a non-request message");
    }
    if (when <= sim_.now()) {
      reply(req, std::move(value));
      return;
    }
    // Hold via channel shared_ptr by wrapping — use inflight if same, else
    // require caller to keep message alive. Prefer channel-owned delayed reply.
    auto channel = req.reply_channel_ptr();
    delayed_reply_channel(sim_, std::move(channel), when, std::move(value));
  }

  simcpp20::process<> run() { return run_loop(sim_, *this); }

  void start() { run_process_ = run(); }

  void stop() {
    running_ = false;
    run_process_.abort();
  }

  simcpp20::simulation<> &sim() noexcept { return sim_; }

  bool has_error() const noexcept { return static_cast<bool>(error_); }

  void rethrow_if_error() const {
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

private:
  static simcpp20::process<>
  delayed_deliver(simcpp20::simulation<> &sim, dynamic_actor &self, double when,
                  std::shared_ptr<dynamic_message> msg) {
    co_await sim.timeout(when - sim.now());
    self.send(std::move(msg));
  }

  static simcpp20::process<>
  delayed_reply(simcpp20::simulation<> &sim,
                std::shared_ptr<dynamic_message> msg, double when,
                std::any value) {
    co_await sim.timeout(when - sim.now());
    if (msg && msg->is_request()) {
      msg->reply_channel()->complete(std::move(value));
    }
  }

  static simcpp20::process<>
  delayed_reply_channel(simcpp20::simulation<> &sim,
                        std::shared_ptr<dynamic_reply_channel> channel,
                        double when, std::any value) {
    co_await sim.timeout(when - sim.now());
    if (channel) {
      channel->complete(std::move(value));
    }
  }

  static simcpp20::process<>
  run_loop(simcpp20::simulation<> &sim, dynamic_actor &self) {
    while (self.running_) {
      try {
        auto msg = co_await self.mailbox_.get();
        auto it = self.handlers_.find(msg->type_id());
        if (it == self.handlers_.end()) {
          throw std::runtime_error("unhandled message type: " +
                                   message_registry::instance().name(
                                       msg->type_id()));
        }
        self.inflight_request_ = msg->is_request() ? msg.get() : nullptr;
        self.inflight_message_ = msg;
        co_await it->second(sim, self, msg);
        if (msg->is_request() && msg->reply_channel() &&
            !msg->reply_channel()->replied) {
          msg->reply_channel()->complete(std::any{});
        }
        self.inflight_request_ = nullptr;
        self.inflight_message_.reset();
      } catch (...) {
        self.inflight_request_ = nullptr;
        self.inflight_message_.reset();
        self.error_ = std::current_exception();
        self.running_ = false;
        co_return;
      }
    }
  }

  simcpp20::simulation<> &sim_;
  simcpp20::store<std::shared_ptr<dynamic_message>> mailbox_;
  std::unordered_map<std::size_t, dynamic_handler_dispatcher> handlers_;
  simcpp20::process<> run_process_;
  bool running_ = true;
  std::exception_ptr error_;
  dynamic_message *inflight_request_ = nullptr;
  std::shared_ptr<dynamic_message> inflight_message_;
};

namespace detail {

template <typename F>
dynamic_handler_dispatcher make_dynamic_dispatcher(F handler) {
  return [handler = std::move(handler)](
             simcpp20::simulation<> &sim, dynamic_actor &self,
             std::shared_ptr<dynamic_message> msg) -> simcpp20::process<> {
    co_await handler(sim, self, std::move(msg));
  };
}

inline simcpp20::process<>
dispatch_sync_dynamic(simcpp20::simulation<> &, dynamic_actor &self,
                      std::shared_ptr<dynamic_message> msg,
                      std::function<void(dynamic_actor &,
                                         std::shared_ptr<dynamic_message>)>
                          handler) {
  handler(self, msg);
  co_return;
}

} // namespace detail

} // namespace hybridsim
