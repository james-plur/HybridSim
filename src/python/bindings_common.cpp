#include "bindings_common.hpp"

namespace hybridsim::python {

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
