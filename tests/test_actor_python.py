import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))

import hybridsim_py as hs


def test_sync_handler():
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
    assert counter["value"] == 5


def test_class_message():
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
    assert seen["value"] == 99


def test_multi_type_and_order():
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
    assert order == [("A", 1), ("B", 2), ("A", 3)]


def test_unknown_message_raises():
    sim = hs.Simulation()
    Known = sim.register_message("Known")
    Unknown = sim.register_message("Unknown")
    actor = hs.Actor(sim)
    actor.on(Known, lambda _actor, _msg: None)
    actor.start()
    actor.send(Unknown)
    sim.run()
    assert actor.has_error()
    try:
        actor.check_error()
        assert False, "expected exception"
    except RuntimeError as exc:
        assert "unhandled message type" in str(exc)


def test_ping_pong():
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
    assert rounds["value"] == max_rounds


def test_simulation_run_until():
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
    assert processed >= 1
    assert seen["count"] >= 1


def main():
    test_sync_handler()
    test_class_message()
    test_multi_type_and_order()
    test_unknown_message_raises()
    test_ping_pong()
    test_simulation_run_until()
    print("All Python actor tests passed.")


if __name__ == "__main__":
    main()
