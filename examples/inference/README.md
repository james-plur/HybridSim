# Inference simulation examples (hybridsim_infer)

Native Actor-based inference skeleton on hybridsim. Corresponds to
`hybridsimdesign/基于actor系统的推理仿真设计.md`, **NO_NETWORK** phase.

## Layout vs design doc

| Design | Package |
|--------|---------|
| ClusterSchedulerActor | `hybridsim_infer.actors.ClusterSchedulerActor` |
| ReplicaSchedulerActor | `hybridsim_infer.actors.ReplicaSchedulerActor` |
| WorkerEngine | `hybridsim_infer.actors.WorkerEngine` (wraps `EngineActor`) |
| Request / Msg types | `hybridsim_infer.request`, `hybridsim_infer.messages` |
| KvCacheManager | `hybridsim_infer.kv_cache` (stub) |
| `dispatch` / wait / running / `batch` / workload | `hybridsim_infer.stubs` (dummy + `# TODO`) |
| Topology assembly | `hybridsim_infer.builder.build_inference_simulation` |

**Not in this phase:** Network actor, live KV Store/Client path (stub actors exist but are not wired), Frontier estimator / multi-kernel DAG.

## Run

From the hybridsim repo root (after `pip install -e .`):

```bash
python examples/inference/monolithic_demo.py
python -m unittest tests.test_inference_skeleton
```

## Minimal API

```python
from hybridsim_infer import InferenceConfig, InferenceRequest, build_inference_simulation

sim = build_inference_simulation(InferenceConfig(num_replicas=1, step_interval=0.001))
sim.schedule_arrivals([
    InferenceRequest(request_id=1, arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=8),
])
sim.run()
assert len(sim.finished_requests) == 1
```

## Step loop

`ReplicaSchedulerActor` kicks `StepMsg` on start / new request / batch end.
After each step it reschedules with `delay=step_interval` while there is work;
when waiting/running/inflight are empty it stops (no zero-time busy loop).

## Relation to Frontier examples

`examples/frontier` remains the Frontier-aligned reference. This package does
**not** import Frontier; replace stub bodies when wiring real schedule logic
(see also `hybridsimdesign/vLLM Engine schedule 逐行注释分析.md`).
