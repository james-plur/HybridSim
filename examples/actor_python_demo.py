import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))

import hybridsim_py as hs


def main():
    sim = hs.Simulation()
    Ping = sim.register_message("Ping")
    Pong = sim.register_message("Pong")

    ping = hs.Actor(sim)
    pong = hs.Actor(sim)
    rounds = {"value": 0}
    max_rounds = 3

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

    print(f"completed {rounds['value']} rounds")


if __name__ == "__main__":
    main()
