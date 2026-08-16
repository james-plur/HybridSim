#pragma once

#include "hybridsim/message.hpp"

#include "fschuetz04/simcpp20.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace hybridsim {

/**
 * simcpp20::store protocol with ``Levels`` FIFO bands and strict priority.
 * Priority ``1`` is served first; ``Levels`` last. Same band keeps FIFO order.
 */
template <typename T, int Levels = kMsgPriorityLevels, typename Time = double>
class priority_store {
  static_assert(Levels >= 1, "priority_store needs at least one level");

public:
  explicit priority_store(
      simcpp20::simulation<Time> &sim,
      size_t capacity = std::numeric_limits<size_t>::max())
      : sim_{sim}, capacity_{capacity} {}

  simcpp20::value_event<T, Time> get() {
    auto ev = sim_.template event<T>();
    ev.add_callback([this]() { trigger_puts(); });
    gets_.push(ev);
    trigger_gets();
    return ev;
  }

  simcpp20::event<Time> put(const T &value,
                            int priority = kMsgPriorityDefault) {
    T copy = value;
    return put(std::move(copy), priority);
  }

  simcpp20::event<Time> put(T &&value, int priority = kMsgPriorityDefault) {
    check_priority(priority);
    auto ev = sim_.event();
    ev.add_callback([this]() { trigger_gets(); });
    puts_.emplace(ev, std::move(value), priority);
    trigger_puts();
    return ev;
  }

private:
  simcpp20::simulation<Time> &sim_;
  std::array<std::queue<T>, static_cast<size_t>(Levels)> values_{};
  size_t capacity_;
  std::queue<simcpp20::value_event<T, Time>> gets_;
  std::queue<std::tuple<simcpp20::event<Time>, T, int>> puts_;

  static void check_priority(int priority) {
    if (priority < 1 || priority > Levels) {
      throw std::runtime_error("invalid message priority " +
                               std::to_string(priority) + " (expected 1.." +
                               std::to_string(Levels) + ")");
    }
  }

  size_t size() const {
    size_t n = 0;
    for (const auto &q : values_) {
      n += q.size();
    }
    return n;
  }

  bool empty() const {
    for (const auto &q : values_) {
      if (!q.empty()) {
        return false;
      }
    }
    return true;
  }

  T pop_highest() {
    for (auto &q : values_) {
      if (!q.empty()) {
        T value = std::move(q.front());
        q.pop();
        return value;
      }
    }
    throw std::logic_error("priority_store::pop_highest on empty store");
  }

  void trigger_gets() {
    while (!empty() && !gets_.empty()) {
      auto ev = std::move(gets_.front());
      gets_.pop();
      if (ev.aborted()) {
        continue;
      }
      ev.trigger(pop_highest());
    }
  }

  void trigger_puts() {
    while (size() < capacity_ && !puts_.empty()) {
      auto [ev, value, priority] = std::move(puts_.front());
      puts_.pop();
      if (ev.aborted()) {
        continue;
      }
      values_[static_cast<size_t>(priority - 1)].emplace(std::move(value));
      ev.trigger();
    }
  }
};

} // namespace hybridsim
