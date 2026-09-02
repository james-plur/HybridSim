# 仿真输出

本文说明推理仿真**读什么结果**、**怎么配置落盘**。静态配置见 [inference_config.md](inference_config.md) 的 `output` 组；全景见 [architecture.md](architecture.md)。

---

## 1. 内存结果（每次 `run()` 后）

`build_inference_simulation` 返回 [`InferenceSimulation`](../src/python/hybridsim_infer/builder.py) 句柄，运行结束后在内存中读取：

| API | 内容 |
|-----|------|
| `infra.finished_requests` | 已完成请求的 `InferenceRequest` 列表（含 `finished_at`、`prefix_hit_tokens` 等） |
| `infra.metrics()` | 聚合 dict（见下表） |
| `infra.now` | 当前 DES 时间（秒） |
| `infra.cluster.arrived_count` | 已调度到达的请求数（可与 `n_finished` 对照） |

`metrics()` 字段（由 [`summarize_metrics`](../src/python/hybridsim_infer/results.py) 计算）：

| 字段 | 含义 |
|------|------|
| `mean_ttft_s` | 已完成请求的平均 TTFT（秒）；无完成请求时为 `null` |
| `tps` | prefill token 数 / 完成请求时间跨度 |
| `hit_rate` | Prefill 阶段 APC ∪ Store 命中 token / `prefill_tokens`；**不含** PD Decode 拉 KV |
| `n_finished` | 已完成请求数 |
| `n_scheduled` | 已到达并分发的请求数 |
| `sim_now_s` | 仿真终态时间 |
| `prefill_tokens` | 已完成请求的 prefill token 总和 |
| `prefix_hit_tokens` | 各请求最长 APC/Store 前缀命中之和（同样不含 PD 传输） |

```python
from hybridsim_infer import InferenceConfig, build_inference_simulation

cfg = InferenceConfig()
infra = build_inference_simulation(cfg)
infra.schedule_arrivals(requests)
infra.run()

print(infra.metrics())
for req in infra.finished_requests:
    print(req.request_id, req.finished_at)
infra.check_errors()  # 聚合 Python / C++ Actor 异常
```

`InferenceSimulation` 还暴露 `sim`、`cluster`、`replicas`、`config`、`kv_store`（若启用）、`profile_path`（若开 trace），供测试或扩展挂接。

---

## 2. 文件产物（`output` 配置）

由 [`OutputConfig`](../src/python/hybridsim_infer/config/output.py) 控制，**默认全部关闭**。路径解析规则：

1. 各子项显式 `path` 优先；
2. 否则 `output.dir` + 默认文件名；
3. 若 `dir` 也未设，写到当前工作目录下的默认文件名。

`run()` 的 `finally` 里会自动调用 `write_outputs()`；也可在 `run()` 前手动调用。返回值 `{ "metrics": Path, "requests": Path, ... }` 只含本次实际写入的项。

| 配置开关 | 默认文件名 | 内容 |
|----------|------------|------|
| `output.metrics.enabled` | `metrics.json` | `metrics()` 的 JSON |
| `output.requests.enabled` | `requests.jsonl` | 每条请求一行（[`request_record`](../src/python/hybridsim_infer/results.py)） |
| `output.config_snapshot.enabled` | `config.json` | 本次 `InferenceConfig` 快照 |
| `output.request_profile.enabled` | `request_profile.json`（或 `dir` 下） | Chrome Trace 风格时间线 |

### `requests.jsonl` 每行字段

| 字段 | 含义 |
|------|------|
| `request_id` | 请求 ID |
| `arrived_at` / `finished_at` | 到达 / 完成仿真时间（秒） |
| `num_prefill_tokens` / `num_decode_tokens` | 请求形状 |
| `num_computed_tokens` / `num_output_tokens` | 推进进度 |
| `prefix_hit_tokens` | Prefill APC/Store 命中 token 数（不含 PD Decode 拉 KV） |
| `completed` / `status` | 是否完成、状态枚举名 |

### 配置示例

```python
from pathlib import Path
from hybridsim_infer import (
    InferenceConfig,
    OutputConfig,
    ArtifactOutput,
    RequestProfileOutput,
)

cfg = InferenceConfig(
    output=OutputConfig(
        dir=Path("out"),
        metrics=ArtifactOutput(enabled=True),
        requests=ArtifactOutput(enabled=True),
        config_snapshot=ArtifactOutput(enabled=True),
        request_profile=RequestProfileOutput(enabled=True),
    ),
)
```

单独指定路径：

```python
OutputConfig(
    metrics=ArtifactOutput(enabled=True, path=Path("results/metrics.json")),
    requests=ArtifactOutput(enabled=True, path=Path("results/requests.jsonl")),
)
```

---

## 3. Request profile（Chrome Trace）

`output.request_profile` 开启时，仿真在**子进程**收集 schedule / engine / KV 事件，写出 JSON。`run()` 结束后可通过 `infra.profile_path` 拿到文件路径。

轨道与打开方式见 [examples/inference/README.md](../examples/inference/README.md) 的 Request profile 一节；嵌套字段说明见 [inference_config.md](inference_config.md) 的 `output.request_profile`。
