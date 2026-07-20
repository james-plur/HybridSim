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

  void send(std::shared_ptr<dynamic_message> msg) {
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
        co_await it->second(sim, self, msg);
      } catch (...) {
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
