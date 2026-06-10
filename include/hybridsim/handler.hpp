#pragma once

#include "hybridsim/message.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <functional>
#include <type_traits>

namespace hybridsim {

class actor;

template <typename F, typename Msg>
inline constexpr bool is_sync_handler_v =
    std::is_invocable_r_v<void, F, actor &, Msg &>;

template <typename F, typename Msg>
inline constexpr bool is_async_handler_v = std::is_invocable_r_v<
    simcpp20::process<>, F, simcpp20::simulation<> &, actor &, Msg &>;

template <typename F, typename Msg>
inline constexpr bool is_valid_handler_v =
    is_sync_handler_v<F, Msg> || is_async_handler_v<F, Msg>;

using handler_dispatcher = std::function<simcpp20::process<>(
    simcpp20::simulation<> &, actor &, std::shared_ptr<message>)>;

namespace detail {

template <typename Msg, typename F>
simcpp20::process<> dispatch_sync(simcpp20::simulation<> &,
                                  actor &self, std::shared_ptr<message> msg,
                                  F handler) {
  handler(self, as_message<Msg>(*msg));
  co_return;
}

template <typename Msg, typename F>
simcpp20::process<> dispatch_async(simcpp20::simulation<> &sim, actor &self,
                                   std::shared_ptr<message> msg, F handler) {
  co_await handler(sim, self, as_message<Msg>(*msg));
}

template <typename Msg, typename F>
handler_dispatcher make_dispatcher(F handler) {
  static_assert(is_valid_handler_v<F, Msg>,
                "handler must be void(actor&, Msg&) or "
                "process<>(simulation&, actor&, Msg&)");

  if constexpr (is_async_handler_v<F, Msg>) {
    return [handler = std::move(handler)](
               simcpp20::simulation<> &sim, actor &self,
               std::shared_ptr<message> msg) -> simcpp20::process<> {
      return dispatch_async<Msg>(sim, self, std::move(msg), handler);
    };
  } else {
    return [handler = std::move(handler)](
               simcpp20::simulation<> &sim, actor &self,
               std::shared_ptr<message> msg) -> simcpp20::process<> {
      return dispatch_sync<Msg>(sim, self, std::move(msg), handler);
    };
  }
}

} // namespace detail

} // namespace hybridsim
