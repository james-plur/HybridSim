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

namespace {

bool is_coroutine(const py::handle &obj) {
  return py::module_::import("inspect")
      .attr("iscoroutine")(obj)
      .cast<bool>();
}

std::any py_to_any(py::object value) {
  if (value.is_none()) {
    return std::any{};
  }
  return std::any(std::move(value));
}

simcpp20::process<> drive_python_coroutine(simcpp20::simulation<> & /*sim*/,
                                           py::object coro) {
  py::object send_arg = py::none();
  while (true) {
    py::object yielded;
    try {
      py::gil_scoped_acquire gil;
      yielded = coro.attr("send")(send_arg);
    } catch (py::error_already_set &err) {
      if (err.matches(PyExc_StopIteration)) {
        co_return;
      }
      throw std::runtime_error("python async handler raised an exception");
    }

    // Expect ReplyFuture yielded from ReplyFuture.__await__
    ReplyFuture *fut = nullptr;
    {
      py::gil_scoped_acquire gil;
      if (!py::isinstance<ReplyFuture>(yielded)) {
        throw std::runtime_error(
            "async handler may only await hybridsim ReplyFuture "
            "(from actor.request / request_at)");
      }
      fut = yielded.cast<ReplyFuture *>();
    }
    if (!fut || !fut->channel) {
      throw std::runtime_error("invalid ReplyFuture");
    }
    co_await fut->channel->event;
    {
      py::gil_scoped_acquire gil;
      send_arg = fut->value();
    }
  }
}

simcpp20::process<> invoke_python_handler(
    simcpp20::simulation<> &sim, const std::shared_ptr<PythonActor> &self,
    const py::function &handler, std::shared_ptr<dynamic_message> msg) {
  py::object result;
  {
    py::gil_scoped_acquire gil;
    try {
      py::object py_msg =
          registry().wrap_payload(msg->type_id(), msg->payload());
      result = handler(self, py_msg);
    } catch (py::error_already_set &) {
      throw std::runtime_error("python handler raised an exception");
    }
  }

  if (!result.is_none() && is_coroutine(result)) {
    co_await drive_python_coroutine(sim, std::move(result));
  }
  co_return;
}

} // namespace

void actor_on(const std::shared_ptr<PythonActor> &self,
              const MessageType &msg_type, py::function handler) {
  self->actor->on(msg_type.id,
                  [self, handler = std::move(handler)](
                      simcpp20::simulation<> &sim, dynamic_actor &,
                      std::shared_ptr<dynamic_message> msg)
                      -> simcpp20::process<> {
                    return invoke_python_handler(sim, self, handler,
                                                 std::move(msg));
                  });
}

void actor_send_object(PythonActor &self, py::object obj, double delay) {
  const std::size_t type_id = registry().type_id_for(obj);
  if (!py::hasattr(obj, "_hybridsim_type_id")) {
    obj.attr("_hybridsim_type_id") = py::int_(type_id);
    obj.attr("_hybridsim_type_name") =
        py::str(message_registry::instance().name(type_id));
  }
  self.actor->send(make_dynamic_message(type_id, std::any(py::object(obj))),
                   delay);
}

void actor_send_type(PythonActor &self, const MessageType &msg_type,
                     py::kwargs kwargs, double delay) {
  py::object obj = make_message_object(msg_type, kwargs);
  self.actor->send(make_dynamic_message(msg_type.id, std::any(obj)), delay);
}

void actor_send_at_object(PythonActor &self, double when, py::object obj) {
  const std::size_t type_id = registry().type_id_for(obj);
  if (!py::hasattr(obj, "_hybridsim_type_id")) {
    obj.attr("_hybridsim_type_id") = py::int_(type_id);
    obj.attr("_hybridsim_type_name") =
        py::str(message_registry::instance().name(type_id));
  }
  self.actor->send_at(when,
                      make_dynamic_message(type_id, std::any(py::object(obj))));
}

void actor_send_at_type(PythonActor &self, double when,
                        const MessageType &msg_type, py::kwargs kwargs) {
  py::object obj = make_message_object(msg_type, kwargs);
  self.actor->send_at(when,
                      make_dynamic_message(msg_type.id, std::any(obj)));
}

ReplyFuture actor_request_object(PythonActor &self, py::object obj,
                                 double delay) {
  const std::size_t type_id = registry().type_id_for(obj);
  if (!py::hasattr(obj, "_hybridsim_type_id")) {
    obj.attr("_hybridsim_type_id") = py::int_(type_id);
    obj.attr("_hybridsim_type_name") =
        py::str(message_registry::instance().name(type_id));
  }
  auto msg =
      make_dynamic_request(type_id, std::any(py::object(obj)), *self.state->sim);
  ReplyFuture fut{msg->reply_channel_ptr()};
  self.actor->send(std::move(msg), delay);
  return fut;
}

ReplyFuture actor_request_type(PythonActor &self, const MessageType &msg_type,
                               py::kwargs kwargs, double delay) {
  py::object obj = make_message_object(msg_type, kwargs);
  return actor_request_object(self, std::move(obj), delay);
}

ReplyFuture actor_request_at_object(PythonActor &self, double when,
                                    py::object obj) {
  const std::size_t type_id = registry().type_id_for(obj);
  if (!py::hasattr(obj, "_hybridsim_type_id")) {
    obj.attr("_hybridsim_type_id") = py::int_(type_id);
    obj.attr("_hybridsim_type_name") =
        py::str(message_registry::instance().name(type_id));
  }
  auto msg =
      make_dynamic_request(type_id, std::any(py::object(obj)), *self.state->sim);
  ReplyFuture fut{msg->reply_channel_ptr()};
  self.actor->send_at(when, std::move(msg));
  return fut;
}

ReplyFuture actor_request_at_type(PythonActor &self, double when,
                                  const MessageType &msg_type,
                                  py::kwargs kwargs) {
  py::object obj = make_message_object(msg_type, kwargs);
  return actor_request_at_object(self, when, std::move(obj));
}

void actor_reply(PythonActor &self, py::object value) {
  self.actor->reply(py_to_any(std::move(value)));
}

void actor_reply_at(PythonActor &self, double when, py::object value) {
  self.actor->reply_at(when, py_to_any(std::move(value)));
}

py::object actor_current_request(PythonActor &self) {
  auto *req = self.actor->current_request();
  if (!req) {
    return py::none();
  }
  // Expose payload object for introspection; reply goes through actor.reply().
  return registry().wrap_payload(req->type_id(), req->payload());
}

void check_actor_errors(const PythonActor &self) {
  if (self.actor->has_error()) {
    self.actor->rethrow_if_error();
  }
}

} // namespace hybridsim::python
