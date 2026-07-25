# Inference simulation examples (hybridsim_infer)

Native Actor-based inference on hybridsim. Corresponds to
`hybridsimdesign/基于actor系统的推理仿真设计.md` and
`hybridsimdesign/hybridsim inference offline校准.md` (**NO_NETWORK**).

## Layout vs design doc

| Design | Package |
|--------|---------|
| ClusterSchedulerActor | `hybridsim_infer.actors.ClusterSchedulerActor` |
| ReplicaSchedulerActor | `hybridsim_infer.actors.ReplicaSchedulerActor` |
| WorkerEngine | `hybridsim_infer.actors.WorkerEngine` |
| KV Store / KV Client | `KvStoreActor` + `KvClientEngine`（`enable_kv_client=True`） |
| schedule / batch | `hybridsim_infer.frameworks`（默认 `VllmFramework`；`FrameworkFactory` 可扩展） |
| Fake GPU duration | `hybridsim_infer.predictors`（`fixed` / `token_proportional`） |

## Schedule alignment（测试）

调度对齐套件在 **[`tests/schedule_alignment/`](../../tests/schedule_alignment/)**（不是 example）。

文档：[schedule_alignment.md](../../tests/schedule_alignment/schedule_alignment.md) · 单元测试：`tests/test_schedule_alignment.py`

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## Run demo / tests

```bash
python examples/inference/monolithic_demo.py
PYTHONPATH=src/python:. python tests/test_inference_skeleton.py -v
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

`InferenceConfig(duration_mode="token_proportional")` 时 batch 时长 ∝ prefill/decode token 数。

`InferenceConfig(framework="vllm")` 选择 replica 内调度实现；扩展时实现 `InferenceFramework` 子类并 `FrameworkFactory.register("sglang", …)`。

## Relation to Frontier

`examples/frontier` 仍是 Frontier 对齐参考；本包不依赖 Frontier。
