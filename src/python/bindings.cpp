#include "hybridsim/dynamic_actor.hpp"
#include "hybridsim/dynamic_message.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
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

TypeRegistry &registry() {
  static TypeRegistry instance;
  return instance;
}

py::object make_message_object(const MessageType &msg_type, py::kwargs kwargs) {
  py::object obj;
  if (!msg_type.py_class.is_none()) {
    obj = msg_type.py_class(**kwargs);
  } else {
    obj = py::module_::import("types").attr("SimpleNamespace")(**kwargs);
  }
  obj.attr("_hybridsim_type_id") = py::int_(msg_type.id);
  obj.attr("_hybridsim_type_name") = py::str(msg_type.name);
  return obj;
}

MessageType register_message(py::object spec) {
  if (py::isinstance<py::str>(spec)) {
    const std::string name = spec.cast<std::string>();
    return registry().register_type(name, py::none());
  }
  py::type cls =
      py::isinstance<py::type>(spec) ? spec.cast<py::type>() : py::type::of(spec);
  const std::string name = py::str(cls.attr("__name__")).cast<std::string>();
  return registry().register_type(name, py::object(cls));
}

void actor_on(const std::shared_ptr<PythonActor> &self,
              const MessageType &msg_type, py::function handler) {
  self->actor->on(msg_type.id,
                  [self, handler = std::move(handler)](
                      simcpp20::simulation<> &, dynamic_actor &,
                      std::shared_ptr<dynamic_message> msg) -> simcpp20::process<> {
                    py::gil_scoped_acquire gil;
                    try {
                      py::object py_msg = registry().wrap_payload(
                          msg->type_id(), msg->payload());
                      handler(self, py_msg);
                    } catch (py::error_already_set &) {
                      throw std::runtime_error("python handler raised an exception");
                    }
                    co_return;
                  });
}

void actor_send_object(PythonActor &self, py::object obj) {
  const std::size_t type_id = registry().type_id_for(obj);
  if (!py::hasattr(obj, "_hybridsim_type_id")) {
    obj.attr("_hybridsim_type_id") = py::int_(type_id);
    obj.attr("_hybridsim_type_name") =
        py::str(message_registry::instance().name(type_id));
  }
  self.actor->send(make_dynamic_message(type_id, std::any(py::object(obj))));
}

void actor_send_type(PythonActor &self, const MessageType &msg_type,
                     py::kwargs kwargs) {
  py::object obj = make_message_object(msg_type, kwargs);
  self.actor->send(make_dynamic_message(msg_type.id, std::any(obj)));
}

void check_actor_errors(const PythonActor &self) {
  if (self.actor->has_error()) {
    self.actor->rethrow_if_error();
  }
}

} // namespace hybridsim::python

PYBIND11_MODULE(hybridsim_py, m) {
  m.doc() = "Python bindings for hybridsim actor system";

  py::class_<hybridsim::python::SimulationState,
             std::shared_ptr<hybridsim::python::SimulationState>>(m,
                                                                  "Simulation")
      .def(py::init<>())
      .def(
          "run",
          [](hybridsim::python::SimulationState &self) {
            const auto count = self.sim->run();
            return count;
          },
          "Run simulation until no events remain")
      .def(
          "run_until",
          [](hybridsim::python::SimulationState &self, double target) {
            return self.sim->run_until(target);
          },
          py::arg("target"),
          "Run simulation until target time")
      .def("now", [](const hybridsim::python::SimulationState &self) {
        return self.sim->now();
      })
      .def("empty", [](const hybridsim::python::SimulationState &self) {
        return self.sim->empty();
      })
      .def("step", [](hybridsim::python::SimulationState &self) {
        self.sim->step();
      })
      .def(
          "register_message",
          [](hybridsim::python::SimulationState &,
             py::object spec) { return hybridsim::python::register_message(spec); },
          py::arg("name_or_class"),
          "Register a message type from a name or Python class");

  py::class_<hybridsim::python::MessageType>(m, "MessageType")
      .def_readonly("id", &hybridsim::python::MessageType::id)
      .def_readonly("name", &hybridsim::python::MessageType::name)
      .def("__call__",
           [](const hybridsim::python::MessageType &self, py::kwargs kwargs) {
             return hybridsim::python::make_message_object(self, kwargs);
           })
      .def(
          "__repr__",
          [](const hybridsim::python::MessageType &self) {
            return "<MessageType " + self.name + " id=" + std::to_string(self.id) +
                   ">";
          });

  py::class_<hybridsim::python::PythonActor,
             std::shared_ptr<hybridsim::python::PythonActor>>(m, "Actor")
      .def(py::init([](std::shared_ptr<hybridsim::python::SimulationState> state) {
             auto actor = std::make_shared<hybridsim::python::PythonActor>();
             actor->state = std::move(state);
             actor->actor =
                 std::make_shared<hybridsim::dynamic_actor>(*actor->state->sim);
             return actor;
           }),
           py::keep_alive<1, 2>())
      .def(
          "on",
          [](const std::shared_ptr<hybridsim::python::PythonActor> &self,
             const hybridsim::python::MessageType &msg_type, py::function fn) {
            hybridsim::python::actor_on(self, msg_type, std::move(fn));
          },
          py::arg("message_type"), py::arg("handler"),
          "Register a Python handler for a message type")
      .def(
          "send",
          [](hybridsim::python::PythonActor &self, py::object arg, py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              hybridsim::python::actor_send_type(
                  self, arg.cast<hybridsim::python::MessageType>(), kwargs);
              return;
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            hybridsim::python::actor_send_object(self, std::move(arg));
          },
          py::arg("message"),
          "Send a message instance or construct one from a MessageType")
      .def("start",
           [](std::shared_ptr<hybridsim::python::PythonActor> &self) {
             self->actor->start();
           })
      .def("stop",
           [](std::shared_ptr<hybridsim::python::PythonActor> &self) {
             self->actor->stop();
           })
      .def("has_error",
           [](const std::shared_ptr<hybridsim::python::PythonActor> &self) {
             return self->actor->has_error();
           })
      .def("check_error",
           [](const std::shared_ptr<hybridsim::python::PythonActor> &self) {
             hybridsim::python::check_actor_errors(*self);
           });
}
