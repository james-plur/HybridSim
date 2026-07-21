"""Generic message registration helpers for hybridsim_py.Simulation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Union


def register_message(hs_sim, msg_cls: type) -> Any:
    """Register one message class; return the hybridsim MessageType."""
    return hs_sim.register_message(msg_cls)


def register_messages(
    hs_sim,
    messages: Union[Mapping[str, type], Iterable[type]],
) -> dict[str, Any]:
    """Register message classes on ``hs_sim``.

    ``messages`` may be a name→class mapping or an iterable of classes
    (keys default to ``cls.__name__``).
    """
    if isinstance(messages, Mapping):
        items = list(messages.items())
    else:
        items = [(cls.__name__, cls) for cls in messages]
    return {name: hs_sim.register_message(cls) for name, cls in items}
