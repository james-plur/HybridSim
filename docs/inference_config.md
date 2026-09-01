# InferenceConfig

`hybridsim_infer` 的用户配置是嵌套 dataclass。顶层 [`InferenceConfig`](../src/python/hybridsim_infer/config/__init__.py) 只组合各组；请求到达仍在 `build` 之后 `schedule_arrivals` / `schedule_from_generator` 注入。

```python
from hybridsim_infer import (
    InferenceConfig,
    ClusterConfig,
    KvConfig,
    ModelSpec,
    InferWorkloadConfig,
    BatchLevelConfig,
    KvWorkloadConfig,
    OutputConfig,
    RequestProfileOutput,
    build_inference_simulation,
)

cfg = InferenceConfig(
    cluster=ClusterConfig(type="pd", num_prefill_replicas=1, num_decode_replicas=1),
    kv=KvConfig(enable_store=True, enable_prefix_caching=True, block_size=16),
    model=ModelSpec(preset="llama-3.1-8b"),
    infer_workload=InferWorkloadConfig(
        mode="batch_level",
        batch=BatchLevelConfig(predictor="token_proportional"),
    ),
    kv_workload=KvWorkloadConfig(bandwidth_gbps=100.0),
    output=OutputConfig(request_profile=RequestProfileOutput(enabled=True)),
)
infra = build_inference_simulation(cfg)
infra.schedule_arrivals(requests)
infra.run()
```

未写出的组走 `default_factory`。`build_inference_simulation` 会先 `validate()`。

## 分组

| 组 | 类型 | 内容 |
|---|---|---|
| `cluster` | `ClusterConfig` | 拓扑：`type`（`monolith` / `pd`）、`num_replicas` 或 P/D 池大小 |
| `schedule` | `ScheduleConfig` | `cluster.policy`（现仅 `least_load`）；`replica`（`name` + vLLM knobs）；`engine.max_inflight_batches` |
| `kv` | `KvConfig` | GPU 页/容量、APC、`enable_store`、`store` 容量、`lookup` 协议与控制面 RTT |
| `model` | `ModelSpec` | `preset` 或显式 `ModelConfig`（infer DAG 与 KV 体积共用；显式 `config` 覆盖 preset） |
| `infer_workload` | `InferWorkloadConfig` | `mode`：`batch_level`（fixed / token_proportional / frontier）或 `op_level`；`op` 为嵌套的 `OpLevelConfig`（`model` / `parallel` / `device` / `network` / `duration_scale`），默认 `OpLevelConfig()` |

`infer_workload.op` 的类定义仍在 [`workload_generators/configs.py`](../src/python/hybridsim_infer/workload_generators/configs.py)，由 `InferWorkloadConfig` 嵌套持有并从 `hybridsim_infer.config` 再导出。`model.preset` / `model.config` 会在 `InferenceConfig.resolved_op_level()` 时写进 `op.model`（KV 体积与 op-level DAG 共用）。`op.network` 是 collective α-β，与 `kv_workload` 不是同一条链路。
| `kv_workload` | `KvWorkloadConfig` | KV 数据面带宽 / α / 时长下限；与 op-level collective `NetworkConfig` 不是同一条链路 |
| `output` | `OutputConfig` | `request_profile`（Chrome Trace）；可选 `metrics` / `requests` / `config_snapshot`（默认关） |

平台字段 `build_dir` 仍在 `SimulationConfig` 上。已删除未使用的 `step_interval`。

## 最小 monolith

```python
cfg = InferenceConfig(
    cluster=ClusterConfig(num_replicas=1),
    infer_workload=InferWorkloadConfig(
        batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
    ),
)
```

## PD + Store

```python
cfg = InferenceConfig(
    cluster=ClusterConfig(type="pd", num_prefill_replicas=1, num_decode_replicas=1),
    kv=KvConfig(enable_store=True, enable_prefix_caching=True),
    model=ModelSpec(preset="llama-3.1-8b"),
)
```

## 旧扁平字段对照（摘录）

| 旧字段 | 新路径 |
|---|---|
| `cluster_type` | `cluster.type` |
| `num_replicas` | `cluster.num_replicas` |
| `framework` | `schedule.replica.name` |
| `enable_kv_client` | `kv.enable_store` |
| `enable_prefix_caching` | `kv.enable_prefix_caching` |
| `model_preset` | `model.preset` |
| `duration_mode` | `infer_workload.mode` |
| `batch_predictor` | `infer_workload.batch.predictor` |
| `dummy_exec_s` | `infer_workload.batch.fixed.dummy_exec_s` |
| `kv_bandwidth_gbps` | `kv_workload.bandwidth_gbps` |
| `kv_lookup_rtt_s` | `kv.lookup.rtt_s` |
| `enable_request_profile` | `output.request_profile.enabled` |
| `max_inflight_batches` | `schedule.engine.max_inflight_batches` |

请求生成不在 `InferenceConfig` 内，见 [request_generation.md](request_generation.md)。

## 输出

`infra.metrics()` 返回 TTFT / TPS / hit_rate 等聚合。`run()` 结束时若对应 `output.*.enabled` 为真，会写 `metrics.json` / `requests.jsonl` / `config.json`。Request Chrome Trace 仍由 `output.request_profile` 走现有子进程 writer。
