import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))

import hybridsim_py as hs


def test_workload_from_dict():
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
    assert done["id"] == 10
    assert sim.now() == 6.0


def test_workload_spec_class():
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
    assert sim.now() == 8.0


def test_invalid_cycle():
    spec = hs.WorkloadSpec(
        kernels=[
            hs.KernelSpec("A", 0, 1.0, [1]),
            hs.KernelSpec("B", 0, 1.0, [0]),
        ]
    )
    try:
        spec.validate()
        assert False, "expected cycle error"
    except Exception as exc:
        assert "cycle" in str(exc).lower()


def test_kernel_params_extension():
    params = hs.KernelParams.from_dict({"tile_size": 32, "fp16": True, "label": "gemm"})
    assert params.get_int("tile_size") == 32
    assert params.get_bool("fp16") is True
    assert params.get_string("label") == "gemm"
    assert params.to_dict() == {"tile_size": 32, "fp16": True, "label": "gemm"}

    kernel = hs.KernelSpec("gemm", 0, 1.0, [], params)
    assert kernel.params.get_int("tile_size") == 32

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
    assert spec.kernels[0].params.get_int("batch") == 8
    assert spec.kernels[0].params.get_string("dtype") == "fp16"


def main():
    test_workload_from_dict()
    test_workload_spec_class()
    test_invalid_cycle()
    test_kernel_params_extension()
    print("All Python engine tests passed.")


if __name__ == "__main__":
    main()
