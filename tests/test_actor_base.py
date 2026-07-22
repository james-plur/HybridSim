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

    def test_actor_base_request_reply(self) -> None:
        sim = hs.Simulation()

        @dataclass
        class QueryMsg:
            id: int = 0

        @dataclass
        class StartMsg:
            pass

        message_types = {
            "QueryMsg": sim.register_message(QueryMsg),
            "StartMsg": sim.register_message(StartMsg),
        }
        got = {"value": None}

        class Worker(ActorBase):
            @on(QueryMsg)
            def handle_query(self, _actor, msg) -> None:
                self.reply({"echo": msg.id})

        class Client(ActorBase):
            def __init__(self, *, worker: Worker, **kwargs):
                super().__init__(**kwargs)
                self.worker = worker

            @on(StartMsg)
            async def handle_start(self, _actor, _msg):
                got["value"] = await self.request(self.worker, QueryMsg, id=9)

        worker = Worker(sim=sim, hs_actor=hs.Actor(sim), message_types=message_types)
        client = Client(
            worker=worker,
            sim=sim,
            hs_actor=hs.Actor(sim),
            message_types=message_types,
        )
        worker.start()
        client.start()
        client.send(StartMsg)
        sim.run()
        worker.check_error()
        client.check_error()
        self.assertEqual(got["value"], {"echo": 9})

    def test_actor_base_send_request_delay(self) -> None:
        sim = hs.Simulation()

        @dataclass
        class SetMsg:
            value: int = 0

        @dataclass
        class QueryMsg:
            id: int = 0

        @dataclass
        class StartMsg:
            pass

        message_types = {
            "SetMsg": sim.register_message(SetMsg),
            "QueryMsg": sim.register_message(QueryMsg),
            "StartMsg": sim.register_message(StartMsg),
        }
        events: list[tuple[float, str]] = []

        class Worker(ActorBase):
            @on(SetMsg)
            def handle_set(self, _actor, msg) -> None:
                events.append((self.sim.now(), f"set:{msg.value}"))

            @on(QueryMsg)
            def handle_query(self, _actor, msg) -> None:
                events.append((self.sim.now(), f"query:{msg.id}"))
                self.reply({"id": msg.id})

        class Client(ActorBase):
            def __init__(self, *, worker: Worker, **kwargs):
                super().__init__(**kwargs)
                self.worker = worker

            @on(StartMsg)
            async def handle_start(self, _actor, _msg):
                self.worker.send(SetMsg, delay=1.0, value=3)
                reply = await self.request(self.worker, QueryMsg, delay=2.0, id=5)
                events.append((self.sim.now(), f"done:{reply['id']}"))

        worker = Worker(sim=sim, hs_actor=hs.Actor(sim), message_types=message_types)
        client = Client(
            worker=worker,
            sim=sim,
            hs_actor=hs.Actor(sim),
            message_types=message_types,
        )
        worker.start()
        client.start()
        client.send(StartMsg)
        sim.run()
        worker.check_error()
        client.check_error()
        self.assertEqual(
            events,
            [(1.0, "set:3"), (2.0, "query:5"), (2.0, "done:5")],
        )


if __name__ == "__main__":
    unittest.main()
