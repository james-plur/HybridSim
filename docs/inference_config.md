# InferenceConfig

`hybridsim_infer` 的静态配置入口：嵌套 dataclass [`InferenceConfig`](../src/python/hybridsim_infer/config/__init__.py) 描述集群拓扑、调度/KV 策略、workload 估时方式与可选落盘开关。**不**携带具体请求序列；请求在 build 之后注入，见 [request_generation.md](request_generation.md)。输出读取与落盘见 [outputs.md](outputs.md)。

---

## 1. 组装与运行

仿真分三步：**组装** → **注入请求** → **运行并取结果**。

```python
from hybridsim_infer import InferenceConfig, build_inference_simulation
from hybridsim_infer.request_generators import ListRequestGenerator

cfg = InferenceConfig()  # 输入①：静态配置
infra = build_inference_simulation(cfg)  # 校验 + spawn Actor

infra.schedule_from_generator(ListRequestGenerator(...))  # 输入②
# 或 infra.schedule_arrivals(requests)

infra.run()
infra.metrics()
infra.check_errors()
```

| 阶段 | 入口 | 作用 |
|------|------|------|
| 组装 | `build_inference_simulation(config)` | `validate()`，创建 Cluster / Replica / Engine（及可选 Store），返回 `InferenceSimulation` |
| 注入 | `schedule_arrivals` / `schedule_from_generator` | 按 `arrived_at` 投递 `InferenceRequest` |
| 运行 | `run()` | 推进 DES；结束时按 `output` 写可选文件 |
| 读取 | `metrics()`、`finished_requests`、`now` | 见 [outputs.md](outputs.md) |

未写出的组走 `default_factory`。平台字段 `build_dir` 在父类 `SimulationConfig` 上：未 `pip install` 时指向 CMake 构建目录，用于加载 `hybridsim_py`。

---

## 2. 配置组总览

| 配置组 | 控制的仿真行为 | 深入阅读 |
|--------|----------------|----------|
| `cluster` | replica 数量、monolith 或 PD 双池 | [scheduler.md](scheduler.md)、[architecture.md](architecture.md) |
| `schedule` | 集群 least-load；replica token budget / 并发；Engine inflight 槽位 | [scheduler.md](scheduler.md)、[engine.md](engine.md) |
| `kv` | GPU KV 块数、APC、共享 Store、lookup RTT | [kv.md](kv.md) |
| `model` | 模型 preset / `ModelConfig`（op-level 构图与 KV 体积） | [workload_generator.md](workload_generator.md)、[kv.md](kv.md) |
| `infer_workload` | 计算时长：`batch_level` predictor 或 `op_level` mock + Analyzer | [workload_generator.md](workload_generator.md) |
| `kv_workload` | KV pull/push 的 α-β（数据面，非 collective） | [kv.md](kv.md) |
| `output` | metrics / requests / config 快照 / Chrome Trace | [outputs.md](outputs.md) |

`infer_workload.op.device` / `op.network` 是 **Analyzer 解析占位参数**（Roofline / α-β），不是未来的 Computing Platform / Network 仿真器（见 [architecture.md](architecture.md) §2.5）。

---

## 3. `cluster` — `ClusterConfig`

| 字段 | 类型 | 默认 | 作用 |
|------|------|------|------|
| `type` | `str` | `"monolith"` | `"monolith"`：所有 replica 等价；`"pd"`：Prefill / Decode 双池 |
| `num_replicas` | `int` | `1` | monolith 下总 replica 数；`type=pd` 时忽略 |
| `num_prefill_replicas` | `int` | `1` | PD：Prefill 池 replica 数（id `0 .. Np-1`） |
| `num_decode_replicas` | `int` | `1` | PD：Decode 池 replica 数（id `Np .. Np+Nd-1`） |

校验：`type=pd` 时 Prefill 与 Decode 各至少 1 台；monolith 时 `num_replicas >= 1`。

---

## 4. `schedule` — `ScheduleConfig`

### `schedule.cluster` — `ClusterScheduleConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `policy` | `"least_load"` | 集群分发策略；当前仅实现 `least_load` |

### `schedule.replica` — `ReplicaScheduleConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `name` | `"vllm"` | replica 内调度后端（`SchedulerFactory` 名） |
| `tokens_per_step` | `8` | 单请求 prefill chunk 每步最多调度 token 数 |
| `decode_tokens_per_step` | `1` | 每请求每步 decode token 数 |
| `max_num_scheduled_tokens` | `64` | 每步全局 token budget（0 → 不限） |
| `max_num_running_reqs` | `32` | 最大并发 running 请求数 |
| `long_prefill_token_threshold` | `0` | 长 prefill 阈值；0 表示用 `tokens_per_step` |
| `reserve_full_isl` | `True` | 仅当完整序列能放进 KV 时才 admit |

### `schedule.engine` — `EngineConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `max_inflight_batches` | `1` | 每 replica 最大并发 Worker batch 数（占槽至 `BatchEnd`） |

---

## 5. `kv` — `KvConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `block_size` | `16` | GPU KV 页大小（token） |
| `num_gpu_blocks` | `1024` | 本地 GPU KV 容量（页数） |
| `enable_prefix_caching` | `False` | 本地 token 列表前缀缓存（APC） |
| `enable_store` | `False` | 挂载共享 `KvStoreActor` + 每 replica `KvClient` |
| `store.block_size` | `None` | Store 对象大小（token）；`None` 时用 GPU `block_size` |
| `store.num_blocks` | `4096` | Store DRAM 容量（块）；`<=0` 表示不限 |
| `lookup.async_` | `False` | Store 异步 lookup（fire-and-forget + Reply） |
| `lookup.rtt_s` | `1e-3` | lookup / PD 控制面 RTT（秒） |

校验：`store.block_size` 必须是 `block_size` 的正整数倍。

---

## 6. `model` — `ModelSpec`

| 字段 | 作用 |
|------|------|
| `preset` | 预设 id（如 `llama-3.1-8b`、`deepseek-v3`），加载 YAML 为 `ModelConfig` |
| `config` | 显式 `ModelConfig`；设置时覆盖 `preset` |

所有 `infer_workload.mode` 共用：注入 op-level 构图形状与 KV 传输体积。`InferenceConfig.resolved_op_level()` 会把解析后的 `ModelConfig` 合并进 `infer_workload.op.model`。

---

## 7. `infer_workload` — `InferWorkloadConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `mode` | `"batch_level"` | `"batch_level"`：predictor 直接估 batch 时长；`"op_level"`：mock DAG + Analyzer |
| `batch` | `BatchLevelConfig` | `mode=batch_level` 时使用 |
| `op` | `OpLevelConfig` | `mode=op_level` 时使用（亦承载未设 `model` 时的默认 `ModelConfig`） |

### `infer_workload.batch` — `BatchLevelConfig`

| 字段 | 默认 | 作用 |
|------|------|------|
| `predictor` | `"fixed"` | `fixed` / `token_proportional` / `frontier` |
| `fixed.dummy_exec_s` | `0.05` | `predictor=fixed` 时 TimeoutKernel 固定时长 |
| `token_proportional.prefill_s_per_token` | `1e-4` | prefill token 单价 |
| `token_proportional.decode_s_per_token` | `1e-3` | decode token 单价 |
| `token_proportional.base_s` | `0.0` | 固定基底 |
| `frontier.predictor` | `None` | 注入的 Frontier RF predictor（不可序列化） |
| `frontier.cluster_type` | `None` | Frontier `ClusterType`（非 `cluster.type`） |
| `frontier.replica_id` | `0` | RF 用的 replica id |
| `frontier.is_moe` | `False` | MoE 标志 |

### `infer_workload.op` — `OpLevelConfig`

定义在 [`workload_generators/configs.py`](../src/python/hybridsim_infer/workload_generators/configs.py)。

| 字段 | 作用 |
|------|------|
| `model` | Transformer 形状（层数、hidden、MoE、MLA 等）；mock 构图与 KV 体积 |
| `parallel` | TP / PP / EP / DP；`attn_tp_size` / `moe_tp_size` 可分别覆盖 |
| `device` | **Analyzer Roofline**：`peak_flops`、`hbm_bandwidth_bps`、`compute_util`、`hbm_util` |
| `network` | **Analyzer collective α-β**：`alpha_s`、`beta_s_per_byte`（或 `from_bandwidth`） |
| `duration_scale` | 全局时长乘子（离线校准后设置） |

`ModelConfig` 常用字段：`num_layers`、`hidden_size`、`intermediate_size`、`num_q_heads`、`num_kv_heads`、`head_dim`、`dtype_bytes`、`is_moe`、`num_experts`、`attn_variant`（GQA / MLA / DSA 等）、`kv_formula` 等。完整列表见源码 dataclass 注释。

`ParallelConfig`：`tp_size`、`pp_size`、`ep_size`、`dp_size`、`pp_stage`；`resolved_attn_tp()` / `resolved_moe_tp()` 处理分路 TP。

`DeviceConfig`：`effective_peak_flops()` = `peak_flops * compute_util`；`effective_hbm_bandwidth_bps()` 同理。

`NetworkConfig`：集体通信时长 = `alpha_s + beta_s_per_byte * bytes`。

---

## 8. `kv_workload` — `KvWorkloadConfig`

KV 数据面 pull/push 的 TimeoutKernel 时长，与 infer op DAG 中的 collective **无关**。

| 字段 | 默认 | 作用 |
|------|------|------|
| `bandwidth_gbps` | `50.0` | 模拟互联带宽（Gbps） |
| `latency_s` | `0.0` | 固定延迟 α |
| `transfer_s_floor` | `1e-4` | 传输时长下限（秒） |
| `bytes_per_token` | `None` | 无 `model` 时的 bytes/token 回退；优先用 `ModelSpec` |

---

## 9. `output` — `OutputConfig`

落盘开关与路径；默认全关。字段含义、内存 API、`requests.jsonl` 列、Chrome Trace 见 **[outputs.md](outputs.md)**。

| 字段 | 作用 |
|------|------|
| `dir` | 共享输出目录（各子项未设 `path` 时用） |
| `metrics` | `ArtifactOutput(enabled, path)` → `metrics.json` |
| `requests` | → `requests.jsonl` |
| `config_snapshot` | → `config.json` |
| `request_profile` | `RequestProfileOutput(enabled, path, dir)` → Chrome Trace |

---

## 10. 常用组合

**最小可跑**（单 replica、固定 batch 时长）：

```python
from hybridsim_infer import (
    InferenceConfig,
    ClusterConfig,
    InferWorkloadConfig,
    BatchLevelConfig,
    BatchFixedConfig,
)

InferenceConfig(
    cluster=ClusterConfig(num_replicas=1),
    infer_workload=InferWorkloadConfig(
        batch=BatchLevelConfig(fixed=BatchFixedConfig(dummy_exec_s=0.01)),
    ),
)
```

**op-level + 模型 preset**：

```python
from hybridsim_infer import InferenceConfig, ModelSpec, InferWorkloadConfig

InferenceConfig(
    model=ModelSpec(preset="llama-3.1-8b"),
    infer_workload=InferWorkloadConfig(mode="op_level"),
)
```

**PD + Store + 落盘**：

```python
from pathlib import Path
from hybridsim_infer import (
    InferenceConfig,
    ClusterConfig,
    KvConfig,
    OutputConfig,
    ArtifactOutput,
)

InferenceConfig(
    cluster=ClusterConfig(type="pd", num_prefill_replicas=1, num_decode_replicas=1),
    kv=KvConfig(enable_store=True, enable_prefix_caching=True),
    output=OutputConfig(
        dir=Path("out"),
        metrics=ArtifactOutput(enabled=True),
        requests=ArtifactOutput(enabled=True),
    ),
)
```

完整示例见 [examples/inference/README.md](../examples/inference/README.md)。
