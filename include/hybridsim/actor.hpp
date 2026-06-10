#pragma once

#include "hybridsim/handler.hpp"
#include "hybridsim/message.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace hybridsim {

class actor {
public:
  explicit actor(simcpp20::simulation<> &sim)
      : sim_{sim}, mailbox_{sim}, run_process_{sim} {}

  template <typename Msg, typename F> void on(F &&handler) {
    handlers_[typeid(Msg)] = detail::make_dispatcher<Msg>(std::forward<F>(handler));
  }

  void send(std::shared_ptr<message> msg) { mailbox_.put(std::move(msg)); }

  template <typename Msg> void send(Msg msg) {
    send(make_message<Msg>(std::move(msg)));
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

protected:
  simcpp20::simulation<> &sim_;
  simcpp20::store<std::shared_ptr<message>> mailbox_;
  std::unordered_map<std::type_index, handler_dispatcher> handlers_;
  simcpp20::process<> run_process_;
  bool running_ = true;
  std::exception_ptr error_;

private:
  static simcpp20::process<> run_loop(simcpp20::simulation<> &sim, actor &self) {
    while (self.running_) {
      try {
        auto msg = co_await self.mailbox_.get();
        auto it = self.handlers_.find(msg->type());
        if (it == self.handlers_.end()) {
          throw std::runtime_error("unhandled message type: " +
                                   std::string(msg->type().name()));
        }
        co_await it->second(sim, self, msg);
      } catch (...) {
        self.error_ = std::current_exception();
        self.running_ = false;
        co_return;
      }
    }
  }
};

} // namespace hybridsim
