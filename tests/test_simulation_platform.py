"""Platform tests for Simulation actor registry and lifecycle."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from hybridsim import ActorBase, Simulation, SimulationConfig, on


@dataclass
class TickMsg:
    n: int = 0


class SimulationPlatformTests(unittest.TestCase):
    def test_simulation_registers_actor_and_runs(self) -> None:
        events: list[tuple[float, int]] = []

        class Ticker(ActorBase):
            @on(TickMsg)
            def handle_tick(self, _actor, msg) -> None:
                events.append((self.sim.now(), msg.n))

        sim = Simulation(SimulationConfig())
        sim.register_messages([TickMsg])
        ticker = sim.spawn_actor(Ticker)
        sim.before_run = lambda: ticker.send_at(2.0, TickMsg, n=7)
        sim.run()
        sim.check_errors()

        self.assertEqual(events, [(2.0, 7)])
        self.assertEqual(sim.now, 2.0)

    def test_simulation_config_from_cli_args(self) -> None:
        cfg = SimulationConfig.from_cli_args(
            ["--build_dir", "/tmp/build", "--trace_output_dir", "/tmp/trace"]
        )
        self.assertEqual(cfg.build_dir, Path("/tmp/build"))
        self.assertEqual(cfg.trace_output_dir, Path("/tmp/trace"))


if __name__ == "__main__":
    unittest.main()
