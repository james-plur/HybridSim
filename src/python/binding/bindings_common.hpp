#pragma once

#include "hybridsim/dynamic_actor.hpp"
#include "hybridsim/dynamic_message.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace hybridsim::python {

struct SimulationState {
  std::shared_ptr<simcpp20::simulation<>> sim =
      std::make_shared<simcpp20::simulation<>>();
};

struct MessageType {
  std::size_t id = 0;
  std::string name;
  py::object py_class;
};

struct PythonActor {
  std::shared_ptr<SimulationState> state;
  std::shared_ptr<dynamic_actor> actor;
};

/// Awaitable reply handle for Python ``request`` / ``request_at``.
struct ReplyFuture {
  std::shared_ptr<dynamic_reply_channel> channel;

  bool ready() const noexcept {
    return channel && channel->replied;
  }

  py::object value() const {
    if (!ready()) {
      throw std::runtime_error("ReplyFuture is not ready");
    }
    const std::any &raw = channel->event.value();
    if (!raw.has_value()) {
      return py::none();
    }
    return std::any_cast<py::object>(raw);
  }
};

class TypeRegistry {
public:
  MessageType register_type(const std::string &name, py::object py_class) {
    const std::size_t id = message_registry::instance().register_type(name);
    MessageType info{id, name, std::move(py_class)};
    types_by_id_[id] = info;
    if (!info.py_class.is_none()) {
      py_type_to_id_[reinterpret_cast<std::uintptr_t>(info.py_class.ptr())] = id;
    }
    return info;
  }

  std::size_t type_id_for(py::handle obj) const {
    if (py::hasattr(obj, "_hybridsim_type_id")) {
      return obj.attr("_hybridsim_type_id").cast<std::size_t>();
    }
    const auto type_key = reinterpret_cast<std::uintptr_t>(py::type::of(obj).ptr());
    auto it = py_type_to_id_.find(type_key);
    if (it != py_type_to_id_.end()) {
      return it->second;
    }
    throw std::runtime_error("object is not a registered hybridsim message");
  }

  py::object wrap_payload(std::size_t type_id,
                          const std::any &payload) const {
    const auto it = types_by_id_.find(type_id);
    if (it == types_by_id_.end()) {
      throw std::runtime_error("unknown message type id");
    }
    if (payload.has_value()) {
      return std::any_cast<py::object>(payload);
    }
    if (!it->second.py_class.is_none()) {
      return it->second.py_class();
    }
    py::object ns = py::module_::import("types").attr("SimpleNamespace")();
    ns.attr("_hybridsim_type_id") = py::int_(type_id);
    ns.attr("_hybridsim_type_name") = py::str(it->second.name);
    return ns;
  }

private:
  std::unordered_map<std::size_t, MessageType> types_by_id_;
  std::unordered_map<std::uintptr_t, std::size_t> py_type_to_id_;
};

TypeRegistry &registry();

py::object make_message_object(const MessageType &msg_type, py::kwargs kwargs);

MessageType register_message(py::object spec);

void actor_on(const std::shared_ptr<PythonActor> &self,
              const MessageType &msg_type, py::function handler);

void actor_send_object(PythonActor &self, py::object obj, double delay = 0.0);

void actor_send_type(PythonActor &self, const MessageType &msg_type,
                     py::kwargs kwargs, double delay = 0.0);

void actor_send_at_object(PythonActor &self, double when, py::object obj);

void actor_send_at_type(PythonActor &self, double when,
                        const MessageType &msg_type, py::kwargs kwargs);

ReplyFuture actor_request_object(PythonActor &self, py::object obj,
                                 double delay = 0.0);

ReplyFuture actor_request_type(PythonActor &self, const MessageType &msg_type,
                               py::kwargs kwargs, double delay = 0.0);

ReplyFuture actor_request_at_object(PythonActor &self, double when,
                                    py::object obj);

ReplyFuture actor_request_at_type(PythonActor &self, double when,
                                  const MessageType &msg_type,
                                  py::kwargs kwargs);

void actor_reply(PythonActor &self, py::object value);

void actor_reply_at(PythonActor &self, double when, py::object value);

py::object actor_current_request(PythonActor &self);

void check_actor_errors(const PythonActor &self);

} // namespace hybridsim::python
