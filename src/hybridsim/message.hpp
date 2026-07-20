#pragma once

#include <memory>
#include <typeindex>
#include <utility>

namespace hybridsim {

class message {
public:
  virtual ~message() = default;
  virtual std::type_index type() const noexcept = 0;
};

template <typename T>
class typed_message : public message {
public:
  T value;

  explicit typed_message(T v) : value(std::move(v)) {}

  std::type_index type() const noexcept override { return typeid(T); }
};

template <typename T, typename... Args>
std::shared_ptr<message> make_message(Args &&...args) {
  return std::make_shared<typed_message<T>>(T{std::forward<Args>(args)...});
}

template <typename T>
T &as_message(message &msg) {
  return static_cast<typed_message<T> &>(msg).value;
}

template <typename T>
const T &as_message(const message &msg) {
  return static_cast<const typed_message<T> &>(msg).value;
}

} // namespace hybridsim
