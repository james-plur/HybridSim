"""Tests for ActorBase and @on decorator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "build"))

import hybridsim_py as hs

from hybridsim_scheduler.actor_base import ActorBase, on


@dataclass
class PingMsg:
    n: int = 0


@dataclass
class PongMsg:
    n: int = 0


def test_actor_base_on_decorator():
    sim = hs.Simulation()
    message_types = {
        "PingMsg": sim.register_message(PingMsg),
        "PongMsg": sim.register_message(PongMsg),
    }
    events: list[tuple[str, float, int]] = []

    class EchoActor(ActorBase):
        @on(PingMsg)
        def handle_ping(self, _actor, msg) -> None:
            events.append(("ping", self.sim.now(), msg.n))
            self.send_at(1.0, PongMsg, n=msg.n + 1)

        @on(PongMsg)
        def handle_pong(self, _actor, msg) -> None:
            events.append(("pong", self.sim.now(), msg.n))

    actor = EchoActor(
        sim=sim,
        hs_actor=hs.Actor(sim),
        message_types=message_types,
    )
    actor.start()
    actor.send(PingMsg, n=1)
    sim.run()
    actor.check_error()

    assert events == [("ping", 0.0, 1), ("pong", 1.0, 2)]
    assert sim.now() == 1.0


def main():
    test_actor_base_on_decorator()
    print("All ActorBase tests passed.")


if __name__ == "__main__":
    main()
