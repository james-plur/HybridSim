"""Base class and @on decorator for hybridsim scheduler actor wrappers."""

from __future__ import annotations

from typing import Any, Iterator


def on(msg_cls: type):
    """Mark a method as the handler for a message dataclass (e.g. RequestArrivalMsg)."""

    def decorator(fn):
        fn.__actor_msg__ = msg_cls
        return fn

    return decorator


class ActorBase:
    """Thin wrapper around hs.Actor with decorator-based message registration."""

    def __init__(self, *, sim, hs_actor, message_types: dict[str, Any]) -> None:
        self.sim = sim
        self._actor = hs_actor
        self._messages = message_types
        self._bind_handlers()

    def _iter_handler_bindings(self) -> Iterator[tuple[type, str]]:
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for name, obj in cls.__dict__.items():
                if name in seen:
                    continue
                msg_cls = getattr(obj, "__actor_msg__", None)
                if msg_cls is None:
                    continue
                seen.add(name)
                yield msg_cls, name

    def _bind_handlers(self) -> None:
        for msg_cls, method_name in self._iter_handler_bindings():
            msg_type = self._messages[msg_cls.__name__]
            self._actor.on(msg_type, getattr(self, method_name))

    def start(self) -> None:
        self._actor.start()

    def send(self, msg_cls: type, **kwargs) -> None:
        self._actor.send(self._messages[msg_cls.__name__], **kwargs)

    def send_at(self, when: float, msg_cls: type, **kwargs) -> None:
        self._actor.send_at(when, self._messages[msg_cls.__name__], **kwargs)

    def check_error(self) -> None:
        self._actor.check_error()
