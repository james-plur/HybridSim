"""Platform Simulation: actor registry, lifecycle, and DES clock."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from hybridsim.actor_base import ActorBase
from hybridsim.config import SimulationConfig
from hybridsim.messages import register_message, register_messages

ActorT = TypeVar("ActorT", bound=ActorBase)


def _import_hybridsim_py(build_dir: Optional[Path] = None):
    """Import the compiled extension.

    Prefer a normal install (``pip install -e .``). ``build_dir`` is an optional
    escape hatch for a bare CMake ``build/`` tree without installing.
    """
    if build_dir is not None:
        build_pkg = str(Path(build_dir).resolve())
        if build_pkg not in sys.path:
            sys.path.insert(0, build_pkg)
    try:
        import hybridsim_py as hs
    except ImportError as exc:
        raise ImportError(
            "Failed to import hybridsim_py. Install with `pip install -e .` "
            "(builds C++ bindings), or pass SimulationConfig(build_dir=...) "
            "pointing at a CMake build directory that contains hybridsim_py."
        ) from exc
    return hs


class Simulation:
    """Own the DES clock and all registered actors.

    Callers register messages and actors, then ``run()`` (which starts actors),
    optionally ``stop()``, and ``check_errors()``. Only actors expose ``start``.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        if config is None:
            config = SimulationConfig()
        hs = _import_hybridsim_py(config.build_dir)

        self._hs = hs
        self.config = config
        self.hs_sim = hs.Simulation()
        self.message_types: dict[str, Any] = {}
        self._actors: list[ActorBase] = []
        self.before_run: Optional[Callable[[], None]] = None
        self._started = False

    @property
    def sim(self):
        """Underlying ``hybridsim_py.Simulation`` (DES clock)."""
        return self.hs_sim

    @property
    def now(self) -> float:
        return float(self.hs_sim.now())

    def register_message(self, msg_cls: type) -> Any:
        msg_type = register_message(self.hs_sim, msg_cls)
        self.message_types[msg_cls.__name__] = msg_type
        return msg_type

    def register_messages(
        self, messages: Union[dict[str, type], list[type], tuple[type, ...]]
    ) -> dict[str, Any]:
        registered = register_messages(self.hs_sim, messages)
        self.message_types.update(registered)
        return registered

    def add_actor(self, actor: ActorT) -> ActorT:
        self._actors.append(actor)
        return actor

    def spawn_actor(self, actor_cls: type[ActorT], **kwargs) -> ActorT:
        """Construct ``actor_cls`` with a fresh ``hs.Actor`` and register it.

        ``actor_cls`` must accept ``sim=``, ``hs_actor=``, ``message_types=``.
        """
        hs_actor = self._hs.Actor(self.hs_sim)
        actor = actor_cls(
            sim=self.hs_sim,
            hs_actor=hs_actor,
            message_types=self.message_types,
            **kwargs,
        )
        return self.add_actor(actor)

    def create_hs_actor(self):
        """Create a raw ``hybridsim_py.Actor`` bound to this simulation."""
        return self._hs.Actor(self.hs_sim)

    def create_engine_actor(self):
        """Create a raw ``hybridsim_py.EngineActor`` bound to this simulation."""
        return self._hs.EngineActor(self.hs_sim)

    def _start_actors(self) -> None:
        for actor in self._actors:
            actor.start()
        self._started = True

    def run(self) -> None:
        """Start all actors, run optional ``before_run`` hook, then drain DES."""
        if not self._started:
            self._start_actors()
        if self.before_run is not None:
            self.before_run()
        self.hs_sim.run()

    def stop(self) -> None:
        for actor in self._actors:
            actor.stop()
        self._started = False

    def check_errors(self) -> None:
        for actor in self._actors:
            actor.check_error()
