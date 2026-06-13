#include "bindings_common.hpp"
#include "engine_bindings.hpp"

#include "hybridsim/engine/engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace hybridsim::python {

namespace eng = engine;

struct PythonEngineActor {
  std::shared_ptr<SimulationState> state;
  std::shared_ptr<eng::engine_actor> core;
};

namespace {

std::vector<std::size_t>
dependencies_from_python(const py::handle &obj) {
  if (!obj || obj.is_none()) {
    return {};
  }
  std::vector<std::size_t> deps;
  for (const auto &item : obj) {
    deps.push_back(py::cast<std::size_t>(py::reinterpret_borrow<py::object>(item)));
  }
  return deps;
}

void set_param_from_python(eng::kernel_params &params, const std::string &key,
                           const py::handle &value) {
  const py::object obj = py::reinterpret_borrow<py::object>(value);
  if (py::isinstance<py::bool_>(obj)) {
    params.set_bool(key, obj.cast<bool>());
    return;
  }
  if (py::isinstance<py::int_>(obj)) {
    params.set_int(key, obj.cast<int64_t>());
    return;
  }
  if (py::isinstance<py::float_>(obj)) {
    params.set_double(key, obj.cast<double>());
    return;
  }
  if (py::isinstance<py::str>(obj)) {
    params.set_string(key, obj.cast<std::string>());
    return;
  }
  throw std::invalid_argument("kernel param '" + key +
                              "' must be bool, int, float, or str");
}

eng::kernel_params params_from_python(const py::handle &obj) {
  if (!py::isinstance<py::dict>(obj)) {
    throw std::invalid_argument("kernel params must be a dict");
  }
  const py::dict d = py::reinterpret_borrow<py::object>(obj);
  eng::kernel_params params;
  for (const auto &item : d) {
    set_param_from_python(params, py::str(item.first).cast<std::string>(),
                          item.second);
  }
  return params;
}

py::dict params_to_python(const eng::kernel_params &params) {
  py::dict d;
  for (const auto &[key, value] : params.values()) {
    std::visit(
        [&](const auto &v) { d[py::str(key)] = py::cast(v); },
        value);
  }
  return d;
}

eng::kernel_spec kernel_spec_from_python(const py::handle &obj) {
  if (py::isinstance<eng::kernel_spec>(obj)) {
    return obj.cast<eng::kernel_spec>();
  }
  if (!py::isinstance<py::dict>(obj)) {
    throw std::invalid_argument(
        "kernel must be a KernelSpec or dict with name, type, duration");
  }
  const py::dict d = py::reinterpret_borrow<py::object>(obj);
  if (!d.contains("name") || !d.contains("duration")) {
    throw std::invalid_argument("kernel dict requires name and duration");
  }
  return eng::kernel_spec{
      .name = py::str(d["name"]).cast<std::string>(),
      .type = d.contains("type") ? py::int_(d["type"]).cast<int32_t>() : 0,
      .duration = py::float_(d["duration"]).cast<double>(),
      .dependencies = d.contains("dependencies")
                            ? dependencies_from_python(d["dependencies"])
                            : std::vector<std::size_t>{},
      .params = d.contains("params") ? params_from_python(d["params"])
                                     : eng::kernel_params{},
  };
}

eng::workload_spec workload_spec_from_python(const py::handle &obj) {
  if (py::isinstance<eng::workload_spec>(obj)) {
    return obj.cast<eng::workload_spec>();
  }
  if (!py::isinstance<py::dict>(obj)) {
    throw std::invalid_argument(
        "workload must be a WorkloadSpec or dict with workload_id, kernels");
  }
  const py::dict d = py::reinterpret_borrow<py::object>(obj);
  if (!d.contains("kernels")) {
    throw std::invalid_argument("workload dict requires kernels");
  }

  eng::workload_spec spec;
  if (d.contains("workload_id")) {
    spec.workload_id = py::int_(d["workload_id"]).cast<int64_t>();
  }
  for (const auto &item : d["kernels"]) {
    spec.kernels.push_back(
        kernel_spec_from_python(py::reinterpret_borrow<py::object>(item)));
  }
  return spec;
}

} // namespace

void bind_engine(py::module_ &m) {
  py::class_<eng::kernel_params>(m, "KernelParams")
      .def(py::init<>())
      .def("empty", &eng::kernel_params::empty)
      .def("clear", &eng::kernel_params::clear)
      .def("contains", &eng::kernel_params::contains, py::arg("key"))
      .def("set_bool", &eng::kernel_params::set_bool, py::arg("key"),
           py::arg("value"))
      .def("set_int", &eng::kernel_params::set_int, py::arg("key"),
           py::arg("value"))
      .def("set_double", &eng::kernel_params::set_double, py::arg("key"),
           py::arg("value"))
      .def("set_string", &eng::kernel_params::set_string, py::arg("key"),
           py::arg("value"))
      .def(
          "get_bool",
          [](const eng::kernel_params &self, const std::string &key) -> py::object {
            const auto value = self.get_bool(key);
            return value ? py::cast(*value) : py::none();
          },
          py::arg("key"))
      .def(
          "get_int",
          [](const eng::kernel_params &self, const std::string &key) -> py::object {
            const auto value = self.get_int(key);
            return value ? py::cast(*value) : py::none();
          },
          py::arg("key"))
      .def(
          "get_double",
          [](const eng::kernel_params &self, const std::string &key) -> py::object {
            const auto value = self.get_double(key);
            return value ? py::cast(*value) : py::none();
          },
          py::arg("key"))
      .def(
          "get_string",
          [](const eng::kernel_params &self, const std::string &key) -> py::object {
            const auto value = self.get_string(key);
            return value ? py::cast(*value) : py::none();
          },
          py::arg("key"))
      .def("to_dict",
           [](const eng::kernel_params &self) { return params_to_python(self); })
      .def_static("from_dict",
                  [](const py::dict &d) { return params_from_python(d); });

  py::class_<eng::kernel_spec>(m, "KernelSpec")
      .def(py::init<>())
      .def(py::init<std::string, int32_t, double, std::vector<std::size_t>,
                    eng::kernel_params>(),
           py::arg("name"), py::arg("type") = 0, py::arg("duration") = 0.0,
           py::arg("dependencies") = std::vector<std::size_t>{},
           py::arg("params") = eng::kernel_params{})
      .def_readwrite("name", &eng::kernel_spec::name)
      .def_readwrite("type", &eng::kernel_spec::type)
      .def_readwrite("duration", &eng::kernel_spec::duration)
      .def_readwrite("dependencies", &eng::kernel_spec::dependencies)
      .def_readwrite("params", &eng::kernel_spec::params)
      .def("__repr__", [](const eng::kernel_spec &self) {
        return "<KernelSpec name=" + self.name + " type=" +
               std::to_string(self.type) + " duration=" +
               std::to_string(self.duration) + " deps=" +
               std::to_string(self.dependencies.size()) + " params=" +
               std::to_string(self.params.values().size()) + ">";
      });

  py::class_<eng::workload_spec>(m, "WorkloadSpec")
      .def(py::init<>())
      .def(py::init<int64_t, std::vector<eng::kernel_spec>>(), py::arg("workload_id") = 0,
           py::arg("kernels") = std::vector<eng::kernel_spec>{})
      .def_readwrite("workload_id", &eng::workload_spec::workload_id)
      .def_readwrite("kernels", &eng::workload_spec::kernels)
      .def("validate", [](const eng::workload_spec &self) {
        eng::validate_dag(self);
      })
      .def_static("from_dict",
                  [](const py::dict &d) { return workload_spec_from_python(d); })
      .def("__repr__", [](const eng::workload_spec &self) {
        return "<WorkloadSpec id=" + std::to_string(self.workload_id) +
               " kernels=" + std::to_string(self.kernels.size()) + ">";
      });

  py::class_<PythonEngineActor, std::shared_ptr<PythonEngineActor>>(m, "EngineActor")
      .def(py::init([](std::shared_ptr<SimulationState> state) {
             auto wrapper = std::make_shared<PythonEngineActor>();
             wrapper->state = std::move(state);
             wrapper->core =
                 std::make_shared<eng::engine_actor>(*wrapper->state->sim);
             return wrapper;
           }),
           py::keep_alive<1, 2>())
      .def("start",
           [](const std::shared_ptr<PythonEngineActor> &self) {
             self->core->start();
           })
      .def(
          "send_workload",
          [](PythonEngineActor &self, py::object workload) {
            self.core->send(eng::WorkloadMsg{
                .spec = workload_spec_from_python(workload)});
          },
          py::arg("workload"),
          "Submit a WorkloadSpec or dict describing a kernel DAG")
      .def(
          "set_on_workload_complete",
          [](PythonEngineActor &self, py::function handler) {
            self.core->set_on_workload_complete(
                [handler = std::move(handler)](const eng::WorkloadDoneMsg &msg) {
                  py::gil_scoped_acquire gil;
                  handler(py::int_(msg.workload_id));
                });
          },
          py::arg("handler"),
          "Register a callback invoked when a workload finishes")
      .def("has_error",
           [](const std::shared_ptr<PythonEngineActor> &self) {
             return self->core->has_error();
           })
      .def("check_error",
           [](const std::shared_ptr<PythonEngineActor> &self) {
             if (self->core->has_error()) {
               self->core->rethrow_if_error();
             }
           });
}

} // namespace hybridsim::python
