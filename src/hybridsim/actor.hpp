#pragma once

#include "hybridsim/handler.hpp"
#include "hybridsim/message.hpp"
#include "hybridsim/priority_store.hpp"
#include "hybridsim/request.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace hybridsim {

class actor {
public:
  explicit actor(simcpp20::simulation<> &sim)
      : sim_{sim}, mailbox_{sim}, run_process_{sim} {}

  template <typename Msg, typename F> void on(F &&handler) {
    handlers_[typeid(Msg)] =
        detail::make_dispatcher<Msg>(std::forward<F>(handler));
  }

  void send(std::shared_ptr<message> msg, double delay = 0.0,
            int priority = kMsgPriorityDefault) {
    if (delay > 0.0) {
      send_at(sim_.now() + delay, std::move(msg), priority);
      return;
    }
    mailbox_.put(std::move(msg), priority);
  }

  template <typename Msg>
  void send(Msg msg, double delay = 0.0, int priority = kMsgPriorityDefault) {
    send(make_message<Msg>(std::move(msg)), delay, priority);
  }

  // Deliver at simulation time `when` (immediate if when <= now).
  void send_at(double when, std::shared_ptr<message> msg,
               int priority = kMsgPriorityDefault) {
    if (when <= sim_.now()) {
      send(std::move(msg), 0.0, priority);
      return;
    }
    delayed_deliver(sim_, *this, when, std::move(msg), priority);
  }

  template <typename Msg>
  void send_at(double when, Msg msg, int priority = kMsgPriorityDefault) {
    send_at(when, make_message<Msg>(std::move(msg)), priority);
  }

  /// Wrap ``msg`` in a request envelope and return an awaitable reply event.
  /// If ``delay`` > 0, the receiver sees the request after ``delay`` sim time.
  template <typename Reply = std::monostate, typename Msg>
  simcpp20::value_event<Reply>
  request(Msg msg, double delay = 0.0, int priority = kMsgPriorityDefault) {
    auto env =
        std::make_shared<request_envelope<Msg, Reply>>(sim_, std::move(msg));
    auto reply = env->reply_event;
    send(std::static_pointer_cast<message>(env), delay, priority);
    return reply;
  }

  template <typename Reply = std::monostate, typename Msg>
  simcpp20::value_event<Reply>
  request_at(double when, Msg msg, int priority = kMsgPriorityDefault) {
    auto env =
        std::make_shared<request_envelope<Msg, Reply>>(sim_, std::move(msg));
    auto reply = env->reply_event;
    send_at(when, std::static_pointer_cast<message>(env), priority);
    return reply;
  }

  request_base *current_request() const noexcept { return inflight_request_; }

  void reply(request_base &req) { req.reply_default(); }

  template <typename Reply>
  void reply(request_base &req, Reply value) {
    req.reply_any(std::any(std::move(value)));
  }

  void reply_at(double when, request_base &req) {
    if (when <= sim_.now()) {
      reply(req);
      return;
    }
    if (!inflight_message_) {
      throw std::runtime_error("reply_at requires an in-flight request message");
    }
    delayed_reply(sim_, inflight_message_, when, std::any{});
  }

  template <typename Reply>
  void reply_at(double when, request_base &req, Reply value) {
    if (when <= sim_.now()) {
      reply(req, std::move(value));
      return;
    }
    if (!inflight_message_) {
      throw std::runtime_error("reply_at requires an in-flight request message");
    }
    delayed_reply(sim_, inflight_message_, when, std::any(std::move(value)));
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
  priority_store<std::shared_ptr<message>> mailbox_;
  std::unordered_map<std::type_index, handler_dispatcher> handlers_;
  simcpp20::process<> run_process_;
  bool running_ = true;
  std::exception_ptr error_;
  request_base *inflight_request_ = nullptr;
  std::shared_ptr<message> inflight_message_;

private:
  static simcpp20::process<> delayed_deliver(simcpp20::simulation<> &sim,
                                             actor &self, double when,
                                             std::shared_ptr<message> msg,
                                             int priority) {
    co_await sim.timeout(when - sim.now());
    self.send(std::move(msg), 0.0, priority);
  }

  static simcpp20::process<> delayed_reply(simcpp20::simulation<> &sim,
                                           std::shared_ptr<message> msg,
                                           double when, std::any value) {
    co_await sim.timeout(when - sim.now());
    auto *req = dynamic_cast<request_base *>(msg.get());
    if (!req) {
      co_return;
    }
    if (value.has_value()) {
      req->reply_any(std::move(value));
    } else {
      req->reply_default();
    }
  }

  static simcpp20::process<> run_loop(simcpp20::simulation<> &sim, actor &self) {
    while (self.running_) {
      try {
        auto msg = co_await self.mailbox_.get();
        auto *req = dynamic_cast<request_base *>(msg.get());
        const std::type_index lookup =
            req ? req->payload_type() : msg->type();
        auto it = self.handlers_.find(lookup);
        if (it == self.handlers_.end()) {
          throw std::runtime_error(
              "unhandled message type: " +
              std::string(req ? req->payload_type().name()
                              : msg->type().name()));
        }

        self.inflight_request_ = req;
        self.inflight_message_ = msg;
        co_await it->second(sim, self, msg);
        if (req && !req->replied()) {
          req->reply_default();
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
};

} // namespace hybridsim
