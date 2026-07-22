"""Low-level hybridsim_py actor binding tests."""

from __future__ import annotations

import unittest

import hybridsim_py as hs


class ActorPythonTests(unittest.TestCase):
    def test_sync_handler(self) -> None:
        sim = hs.Simulation()
        Increment = sim.register_message("Increment")
        actor = hs.Actor(sim)
        counter = {"value": 0}

        def on_increment(_actor, msg):
            counter["value"] += msg.delta

        actor.on(Increment, on_increment)
        actor.start()
        actor.send(Increment, delta=2)
        actor.send(Increment, delta=3)
        sim.run()
        actor.check_error()
        self.assertEqual(counter["value"], 5)

    def test_class_message(self) -> None:
        sim = hs.Simulation()

        class SetMsgCls:
            def __init__(self, value=0):
                self.value = value

        SetMsg = sim.register_message(SetMsgCls)
        actor = hs.Actor(sim)
        seen = {"value": None}

        def on_set(_actor, msg):
            seen["value"] = msg.value

        actor.on(SetMsg, on_set)
        actor.start()
        actor.send(SetMsgCls(value=99))
        sim.run()
        actor.check_error()
        self.assertEqual(seen["value"], 99)

    def test_multi_type_and_order(self) -> None:
        sim = hs.Simulation()
        A = sim.register_message("A")
        B = sim.register_message("B")
        actor = hs.Actor(sim)
        order = []

        actor.on(A, lambda _actor, msg: order.append(("A", msg.n)))
        actor.on(B, lambda _actor, msg: order.append(("B", msg.n)))
        actor.start()
        actor.send(A, n=1)
        actor.send(B, n=2)
        actor.send(A, n=3)
        sim.run()
        actor.check_error()
        self.assertEqual(order, [("A", 1), ("B", 2), ("A", 3)])

    def test_unknown_message_raises(self) -> None:
        sim = hs.Simulation()
        Known = sim.register_message("Known")
        Unknown = sim.register_message("Unknown")
        actor = hs.Actor(sim)
        actor.on(Known, lambda _actor, _msg: None)
        actor.start()
        actor.send(Unknown)
        sim.run()
        self.assertTrue(actor.has_error())
        with self.assertRaises(RuntimeError) as ctx:
            actor.check_error()
        self.assertIn("unhandled message type", str(ctx.exception))

    def test_ping_pong(self) -> None:
        sim = hs.Simulation()
        Ping = sim.register_message("Ping")
        Pong = sim.register_message("Pong")
        rounds = {"value": 0}
        max_rounds = 3

        ping = hs.Actor(sim)
        pong = hs.Actor(sim)

        def on_ping(_actor, _msg):
            if rounds["value"] < max_rounds:
                pong.send(Pong, round=rounds["value"] + 1)

        def on_pong_ping(actor, msg):
            rounds["value"] = msg.round
            if rounds["value"] < max_rounds:
                actor.send(Ping)

        def on_pong_pong(_actor, msg):
            ping.send(msg)

        ping.on(Ping, on_ping)
        ping.on(Pong, on_pong_ping)
        pong.on(Pong, on_pong_pong)

        ping.start()
        pong.start()
        ping.send(Ping)
        sim.run()
        ping.check_error()
        pong.check_error()
        self.assertEqual(rounds["value"], max_rounds)

    def test_simulation_run_until(self) -> None:
        sim = hs.Simulation()
        Timeout = sim.register_message("Timeout")
        actor = hs.Actor(sim)
        seen = {"count": 0}

        def on_timeout(_actor, _msg):
            seen["count"] += 1

        actor.on(Timeout, on_timeout)
        actor.start()
        actor.send(Timeout)
        actor.send(Timeout)
        processed = sim.run_until(0.5)
        actor.check_error()
        self.assertGreaterEqual(processed, 1)
        self.assertGreaterEqual(seen["count"], 1)

    def test_send_at_delivers_at_time(self) -> None:
        sim = hs.Simulation()
        Tick = sim.register_message("Tick")
        actor = hs.Actor(sim)
        events = []

        def on_tick(_actor, msg):
            events.append((sim.now(), msg.n))

        actor.on(Tick, on_tick)
        actor.start()
        actor.send_at(5.0, Tick, n=2)
        actor.send_at(2.0, Tick, n=1)
        actor.send_at(0.0, Tick, n=0)
        sim.run()
        actor.check_error()
        self.assertEqual(events, [(0.0, 0), (2.0, 1), (5.0, 2)])
        self.assertEqual(sim.now(), 5.0)

    def test_send_at_interleaves_with_immediate(self) -> None:
        sim = hs.Simulation()
        Tick = sim.register_message("Tick")
        actor = hs.Actor(sim)
        events = []

        def on_tick(_actor, msg):
            events.append((sim.now(), msg.label))

        actor.on(Tick, on_tick)
        actor.start()
        actor.send_at(1.0, Tick, label="delayed")
        actor.send(Tick, label="immediate")
        sim.run()
        actor.check_error()
        self.assertEqual(events, [(0.0, "immediate"), (1.0, "delayed")])

    def test_request_explicit_reply(self) -> None:
        sim = hs.Simulation()
        Query = sim.register_message("Query")
        Start = sim.register_message("Start")
        server = hs.Actor(sim)
        client = hs.Actor(sim)
        got = {"value": None}

        def on_query(actor, msg):
            actor.reply({"echo": msg.id})

        async def on_start(actor, _msg):
            fut = server.request(Query, id=7)
            got["value"] = await fut

        server.on(Query, on_query)
        client.on(Start, on_start)
        server.start()
        client.start()
        client.send(Start)
        sim.run()
        server.check_error()
        client.check_error()
        self.assertEqual(got["value"], {"echo": 7})

    def test_request_auto_empty_reply(self) -> None:
        sim = hs.Simulation()
        Query = sim.register_message("Query")
        Start = sim.register_message("Start")
        server = hs.Actor(sim)
        client = hs.Actor(sim)
        got = {"value": "unset"}

        def on_query(_actor, _msg):
            pass  # no reply

        async def on_start(actor, _msg):
            got["value"] = await server.request(Query, id=1)

        server.on(Query, on_query)
        client.on(Start, on_start)
        server.start()
        client.start()
        client.send(Start)
        sim.run()
        client.check_error()
        self.assertIsNone(got["value"])

    def test_request_at(self) -> None:
        sim = hs.Simulation()
        Query = sim.register_message("Query")
        Start = sim.register_message("Start")
        server = hs.Actor(sim)
        client = hs.Actor(sim)
        got = {"at": -1.0, "value": None}

        def on_query(actor, msg):
            actor.reply(msg.id * 2)

        async def on_start(actor, _msg):
            fut = server.request_at(1.5, Query, id=4)
            got["value"] = await fut
            got["at"] = sim.now()

        server.on(Query, on_query)
        client.on(Start, on_start)
        server.start()
        client.start()
        client.send(Start)
        sim.run()
        client.check_error()
        self.assertEqual(got["value"], 8)
        self.assertEqual(got["at"], 1.5)

    def test_send_and_request_delay(self) -> None:
        sim = hs.Simulation()
        Set = sim.register_message("Set")
        Query = sim.register_message("Query")
        Start = sim.register_message("Start")
        server = hs.Actor(sim)
        client = hs.Actor(sim)
        events: list[tuple[float, str]] = []

        def on_set(_actor, msg):
            events.append((sim.now(), f"set:{msg.value}"))

        def on_query(actor, msg):
            events.append((sim.now(), f"query:{msg.id}"))
            actor.reply({"id": msg.id})

        async def on_start(actor, _msg):
            server.send(Set, delay=1.0, value=1)
            reply = await server.request(Query, delay=2.0, id=9)
            events.append((sim.now(), f"client_done:{reply['id']}"))

        server.on(Set, on_set)
        server.on(Query, on_query)
        client.on(Start, on_start)
        server.start()
        client.start()
        client.send(Start)
        sim.run()
        client.check_error()
        server.check_error()
        self.assertEqual(
            events,
            [(1.0, "set:1"), (2.0, "query:9"), (2.0, "client_done:9")],
        )


if __name__ == "__main__":
    unittest.main()
