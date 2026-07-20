#include "bindings_common.hpp"
#include "engine_bindings.hpp"

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
      .def(
          "send_at",
          [](hybridsim::python::PythonActor &self, double when, py::object arg,
             py::kwargs kwargs) {
            if (py::isinstance<hybridsim::python::MessageType>(arg)) {
              hybridsim::python::actor_send_at_type(
                  self, when, arg.cast<hybridsim::python::MessageType>(), kwargs);
              return;
            }
            if (!kwargs.empty()) {
              throw std::runtime_error(
                  "keyword arguments require a MessageType as the first argument");
            }
            hybridsim::python::actor_send_at_object(self, when, std::move(arg));
          },
          py::arg("when"), py::arg("message"),
          "Deliver a message at simulation time `when` (immediate if when <= now)")
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

  hybridsim::python::bind_engine(m);
}
