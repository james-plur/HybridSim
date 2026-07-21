"""Tests for ActorBase and @on decorator."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import hybridsim_py as hs

from hybridsim import ActorBase, on


@dataclass
class PingMsg:
    n: int = 0


@dataclass
class PongMsg:
    n: int = 0


class ActorBaseTests(unittest.TestCase):
    def test_actor_base_on_decorator(self) -> None:
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

        self.assertEqual(events, [("ping", 0.0, 1), ("pong", 1.0, 2)])
        self.assertEqual(sim.now(), 1.0)


if __name__ == "__main__":
    unittest.main()
