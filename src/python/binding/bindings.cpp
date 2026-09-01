#include "bindings_common.hpp"
#include "engine_bindings.hpp"
#include "network_bindings.hpp"

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(hybridsim_py, m) {
  m.doc() = "Python bindings for hybridsim actor system";

  py::class_<hybridsim::python::SimulationState,
             std::shared_ptr<hybridsim::python::SimulationState>>(m,
                                                                  "Simulation")
      .def(py::init<>())
      .def(
          "run",
          [](hybridsim::python::SimulationState &self) {
            return self.sim->run();
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

  py::class_<hybridsim::python::ReplyFuture>(m, "ReplyFuture")
      .def("ready", &hybridsim::python::ReplyFuture::ready)
      .def("value", &hybridsim::python::ReplyFuture::value)
      .def(
          "__repr__",
          [](const hybridsim::python::ReplyFuture &self) {
            return self.ready() ? "<ReplyFuture ready>" : "<ReplyFuture pending>";
          });

  // Make ReplyFuture awaitable: yield self, then return value().
  {
    py::dict scope;
    py::exec(
        R"(
def __await__(self):
    yield self
    return self.value()
)",
        py::globals(), scope);
    m.attr("ReplyFuture").attr("__await__") = scope["__await__"];
  }

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
          [](hybridsim::python::PythonActor &self, py::object arg, double delay,
             int priority, py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              hybridsim::python::actor_send_type(
                  self, arg.cast<hybridsim::python::MessageType>(), kwargs,
                  delay, priority);
              return;
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            hybridsim::python::actor_send_object(self, std::move(arg), delay,
                                                 priority);
          },
          py::arg("message"), py::arg("delay") = 0.0,
          py::arg("priority") = 3,
          "Send a message (optional delay / mailbox priority 1=high .. 5=low)")
      .def(
          "send_at",
          [](hybridsim::python::PythonActor &self, double when, py::object arg,
             int priority, py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              hybridsim::python::actor_send_at_type(
                  self, when, arg.cast<hybridsim::python::MessageType>(),
                  kwargs, priority);
              return;
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            hybridsim::python::actor_send_at_object(self, when, std::move(arg),
                                                    priority);
          },
          py::arg("when"), py::arg("message"), py::arg("priority") = 3,
          "Deliver a message at simulation time `when` (immediate if when <= now)")
      .def(
          "request",
          [](hybridsim::python::PythonActor &self, py::object arg, double delay,
             int priority, py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              return hybridsim::python::actor_request_type(
                  self, arg.cast<hybridsim::python::MessageType>(), kwargs,
                  delay, priority);
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            return hybridsim::python::actor_request_object(
                self, std::move(arg), delay, priority);
          },
          py::arg("message"), py::arg("delay") = 0.0,
          py::arg("priority") = 3,
          "Send a request after optional delay; return ReplyFuture")
      .def(
          "request_at",
          [](hybridsim::python::PythonActor &self, double when, py::object arg,
             int priority, py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              return hybridsim::python::actor_request_at_type(
                  self, when, arg.cast<hybridsim::python::MessageType>(),
                  kwargs, priority);
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            return hybridsim::python::actor_request_at_object(
                self, when, std::move(arg), priority);
          },
          py::arg("when"), py::arg("message"), py::arg("priority") = 3,
          "Deliver a request at `when` and return a ReplyFuture")
      .def(
          "reply",
          [](hybridsim::python::PythonActor &self, py::object value) {
            hybridsim::python::actor_reply(self, std::move(value));
          },
          py::arg("value") = py::none(),
          "Reply to the in-flight request (default None)")
      .def(
          "reply_at",
          [](hybridsim::python::PythonActor &self, double when, py::object value) {
            hybridsim::python::actor_reply_at(self, when, std::move(value));
          },
          py::arg("when"), py::arg("value") = py::none(),
          "Reply to the in-flight request at simulation time `when`")
      .def(
          "current_request",
          [](hybridsim::python::PythonActor &self) {
            return hybridsim::python::actor_current_request(self);
          },
          "Payload of the in-flight request, or None")
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

  hybridsim::python::bind_network(m);
  hybridsim::python::bind_engine(m);
}
