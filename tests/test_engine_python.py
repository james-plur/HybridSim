"""Low-level EngineActor binding tests."""

from __future__ import annotations

import unittest

import hybridsim_py as hs


class EnginePythonTests(unittest.TestCase):
    def test_workload_from_dict(self) -> None:
        sim = hs.Simulation()
        engine = hs.EngineActor(sim)
        done = {"id": None}
        engine.set_on_workload_complete(lambda wid: done.update(id=wid))
        engine.start()

        engine.send_workload(
            {
                "workload_id": 10,
                "kernels": [
                    {"name": "A", "duration": 1.0, "dependencies": []},
                    {"name": "B", "duration": 2.0, "dependencies": [0]},
                    {"name": "C", "duration": 3.0, "dependencies": [1]},
                ],
            }
        )
        sim.run()
        engine.check_error()
        self.assertEqual(done["id"], 10)
        self.assertEqual(sim.now(), 6.0)

    def test_workload_spec_class(self) -> None:
        sim = hs.Simulation()
        engine = hs.EngineActor(sim)
        engine.start()

        workload = hs.WorkloadSpec(
            workload_id=1,
            kernels=[
                hs.KernelSpec("A", 0, 1.0, []),
                hs.KernelSpec("B", 0, 2.0, [0]),
                hs.KernelSpec("C", 0, 3.0, [0]),
                hs.KernelSpec("D", 0, 4.0, [1, 2]),
            ],
        )
        engine.send_workload(workload)
        sim.run()
        engine.check_error()
        self.assertEqual(sim.now(), 8.0)

    def test_invalid_cycle(self) -> None:
        spec = hs.WorkloadSpec(
            kernels=[
                hs.KernelSpec("A", 0, 1.0, [1]),
                hs.KernelSpec("B", 0, 1.0, [0]),
            ]
        )
        with self.assertRaises(Exception) as ctx:
            spec.validate()
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_kernel_params_extension(self) -> None:
        params = hs.KernelParams.from_dict(
            {"tile_size": 32, "fp16": True, "label": "gemm"}
        )
        self.assertEqual(params.get_int("tile_size"), 32)
        self.assertIs(params.get_bool("fp16"), True)
        self.assertEqual(params.get_string("label"), "gemm")
        self.assertEqual(
            params.to_dict(), {"tile_size": 32, "fp16": True, "label": "gemm"}
        )

        kernel = hs.KernelSpec("gemm", 0, 1.0, [], params)
        self.assertEqual(kernel.params.get_int("tile_size"), 32)

        spec = hs.WorkloadSpec.from_dict(
            {
                "kernels": [
                    {
                        "name": "A",
                        "duration": 1.0,
                        "params": {"batch": 8, "dtype": "fp16"},
                    }
                ]
            }
        )
        self.assertEqual(spec.kernels[0].params.get_int("batch"), 8)
        self.assertEqual(spec.kernels[0].params.get_string("dtype"), "fp16")


if __name__ == "__main__":
    unittest.main()
