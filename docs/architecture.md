# hybridsim 推理仿真架构

`hybridsim_infer` 把一个 LLM serving 集群建模成一组 Actor：请求由生成器注入，经集群分发到实例，实例内做调度与 KV 管理，产出的 batch 被翻译成算子 DAG，交给离散事件 Engine「假执行」。

本文是总览，只回答「有哪几层、各层管什么、数据怎么流」。逐层细节在专题文档，见文末阅读顺序。

---

## 1. 全景

```mermaid
flowchart TB
  RG["RequestGenerator<br/>产出 InferenceRequest 序列"]
  CL["ClusterActor + ClusterManager"]
  RP["ReplicaActor"]
  RP2["ReplicaActor ..."]
  KVS["KvStoreActor<br/>(可选共享 KV Store)"]
  ENG["EngineActor<br/>(计算)"]
  KVE["EngineActor<br/>(KV 传输)"]

  RG -->|"RequestArriveMsg"| CL
  CL -->|"RequestMsg"| RP
  CL -.->|"RequestMsg"| RP2
  RP -->|"RequestHandoffMsg / RequestFinishMsg"| CL
  RP <-->|"KVLookupMsg / KVUpdateMsg"| KVS
  RP -->|"workload (kernel DAG)"| ENG
  RP -->|"pull / push workload"| KVE
  ENG -->|"BatchEndMsg"| RP
  KVE -->|"KVTransferEndMsg"| RP

  subgraph timing ["Device / Network：解析模型，非包级仿真"]
    ROOF["Roofline (device)"]
    AB["α-β (network)"]
  end
  timing -.->|"估出 duration"| ENG
```



对照关系：


| 分层               | 代码                                                               | 配置组                                             | 状态                                                 |
| ---------------- | ---------------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| 请求生成             | `hybridsim_infer.request_generators`（List / ServeGen / KV trace） | 不在 config，构建后注入                                 | 已实现                                                |
| 集群分发             | `ClusterActor` + `MonolithClusterManager` / `PdClusterManager`   | `cluster`、`schedule.cluster`                    | monolith / PD + least-load 已实现                     |
| 实例调度             | `ReplicaActor` + `VllmScheduler`（`SchedulerFactory`）             | `schedule.replica`                              | vLLM 语义已实现并对齐                                      |
| KV               | `VllmKvCacheManager`、`KvClient`、`KvStoreActor`                   | `kv`、`kv_workload`                              | 本地 APC + Store 骨架已实现                               |
| workload         | `workload_generators`（`batch_level` / `op_level`）                | `model`、`infer_workload`                        | 已实现                                                |
| Engine 执行        | `WorkerEngine` + C++ `engine_actor`                              | `schedule.engine`                               | kernel DAG 已实现                                     |
| Device / Network | `DeviceConfig` / `NetworkConfig`（解析公式）；可选 `network_sim` 流级仿真     | `infer_workload.op`、`kv_workload`、`network_sim` | 默认解析模型；`network_sim.enabled` 走 C++ 数据面 + Python 拓扑 |


组装入口是 `[build_inference_simulation](../src/python/hybridsim_infer/builder.py)`：读一份嵌套 `[InferenceConfig](../src/python/hybridsim_infer/config/__init__.py)`，spawn 出 Cluster、N 个同构 Replica（默认各带 1 个计算 Engine；`network_sim.enabled` 时每 rank 一个）、启用 Store 时再加传输 Engine、可选共享 Store 与 Python 组网的 C++ fabric，返回 `InferenceSimulation` 句柄。Actor 的配置输入一律是这份 `config`；`replica_id` / `cluster` / `engine` / `kv_store` 等是运行时接线，不再把扁平 knobs 拆进构造参数。

```python
from hybridsim_infer import InferenceConfig, build_inference_simulation

infra = build_inference_simulation(InferenceConfig())
infra.schedule_from_generator(gen)
infra.run()
```

---



## 2. 各层职责



### 2.1 请求生成

- **职责**：决定「什么时候来多少请求、每条多长、前缀怎么共享」。
- **输入 / 输出**：外部 trace 或分布参数 → `list[InferenceRequest]`（带 `arrived_at` / `num_prefill_tokens` / `num_decode_tokens` / 可选 `hash_ids`）。
- **入口**：`ListRequestGenerator`、`ServeGenRequestGenerator`、`KvCacheTraceRequestGenerator`；经 `infra.schedule_arrivals` 或 `schedule_from_generator` 注入。
- **不做**：不决定执行时长，也不进 `InferenceConfig`（构建之后再注入）。

详见 [request_generation.md](request_generation.md)。

### 2.2 集群分发

- **职责**：请求在实例之间怎么放。当前只有 least-load，一个策略两种拓扑。
- **输入 / 输出**：`RequestArriveMsg` / `RequestHandoffMsg` / `RequestFinishMsg` → 向目标 replica 发 `RequestMsg`。
- **入口**：`[actors/cluster.py](../src/python/hybridsim_infer/actors/cluster.py)`、`[cluster/](../src/python/hybridsim_infer/cluster/)`。

两种拓扑：


| `cluster.type` | 到达时选谁                   | 请求上盖的标志                                                                                                    |
| -------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `monolith`     | 全部 replica 里 least-load | 清掉 PD 标志                                                                                                   |
| `pd`           | Prefill 池 least-load    | `do_remote_decode=True`；prefill 算完后 handoff 再从 Decode 池选一台，盖 `do_remote_prefill` + `remote_replica_id=源 P` |


关键约定：**Replica 是同构的**，没有「P 实例代码路径」和「D 实例代码路径」，角色只体现在请求的 `kv_transfer_params` 上。PD 的 handoff 由 `ReplicaActor._maybe_handoff_prefill` 触发，Decode 侧 `num_computed_tokens` 清零后重新拉 KV。

- **不做**：AF（Attention / FFN）分离等其它实例间拓扑尚未实现；第二种分发策略（`schedule.cluster.policy` 只接受 `least_load`）也未实现。



### 2.3 实例内部

一台 replica 每拍做三件事，对应框架图里 replica 的三个过程：

```mermaid
flowchart LR
  Q["waiting / running 队列"] --> S["1. schedule<br/>VllmScheduler.schedule_step"]
  S <--> K["2. kv<br/>allocate / prefix / lookup"]
  S --> B["ScheduleBatch"]
  B --> W["3. workload generator<br/>batch → kernel DAG"]
  W --> E["WorkerEngine.submit"]
```



1. **schedule**：在 token budget、并发上限、KV 容量三重约束下，决定本步哪些请求各算多少 token，产出 `ScheduleBatch`。抢占按 vLLM 的 FCFS 语义。
2. **kv**：调度过程中触发。本地 `allocate` / prefix 命中直接抬高 `num_computed_tokens`；命中远端时排队 pull，请求进入 `WAIT_FOR_REMOTE_KVS`，等 `KVTransferEndMsg` 解锁。
3. **workload generator**：把 `ScheduleBatch` 变成 Engine 能跑的 kernel DAG（`batch_level` 一个 kernel，`op_level` 一整张算子图）。

详见 [scheduler.md](scheduler.md)、[kv.md](kv.md)。

### 2.4 Engine

- **职责**：按依赖关系执行 kernel DAG，推进仿真时钟。
- **输入 / 输出**：workload（kernel 列表 + 依赖）→ 完成回调 → `BatchEndMsg` / `KVTransferEndMsg`。
- **入口**：`[actors/worker_engine.py](../src/python/hybridsim_infer/actors/worker_engine.py)`、`[engine/engine_actor.hpp](../src/hybridsim/engine/engine_actor.hpp)`。
- **不做**：不估计算时长。默认 kernel 是 `TimeoutKernel`。开启 `network_sim` 后通信改为 Put/Wait 等，见 [network.md](network.md)。

详见 [engine.md](engine.md)。

### 2.5 Device / Network

计算时长仍是 Roofline（`DeviceConfig`）。通信有两条路径：

- **默认（NO_NETWORK）**：α-β（`NetworkConfig.alpha_s` / `beta_s_per_byte`），在 workload generator 阶段算完，Engine 只 `timeout`。
- `network_sim.enabled`：通信用 `RingCommAnalyzer` 把 `CommOp` 拆成 Put/Wait；计算仍是 `AnalyticAnalyzer`。每个 rank 一个 Engine，消息进 C++ NetworkAdapter。拓扑与路由表由 Python 插件初始化。详见 [network.md](network.md)。

KV 传输仍走 `kv_workload` α-β，不接入该 fabric。

---



## 3. 一次请求的生命周期

**Monolith**：

```text
arrived_at 到点 → ClusterActor.on_request_arrive → least-load 选 replica → RequestMsg
  → ReplicaActor.on_request：入 waiting，_arm_step
  → on_step：schedule_step 产出 ScheduleBatch（可能只是 prompt 的一个 chunk）
  → workload generator → WorkerEngine.submit → EngineActor 跑 kernel DAG
  → BatchEndMsg → on_batch_complete：推进 computed / output token
      ├─ 未完成 → 继续下一拍
      └─ 完成 → 释放 KV → RequestFinishMsg → Cluster 记账
```

**PD 分离**多两段：prefill 侧算完 prompt 后不继续 decode，而是发 `RequestHandoffMsg`；Cluster 从 Decode 池选一台重发 `RequestMsg`；Decode 侧先走控制面 lookup + KV pull（`WAIT_FOR_REMOTE_KVS`），KV 到齐后才开始 decode。开启 prefix caching 时，Prefill 在 handoff 前会把已算前缀发布到本地 APC，供同 prompt 的后续请求命中。

---



## 4. 平台底座

推理逻辑全在 Python，DES 内核在 C++：

- `[Simulation](../src/python/hybridsim/simulation.py)`：持有 simcpp20 仿真器，`spawn_actor` / `create_engine_actor` / `register_messages` / `run`。
- `ActorBase` + `@on(Msg)`：Python 侧 Actor 基类，`send` / `send_at` / `request` 对应 DES 事件投递；handler 可以是 `async def`，挂起的是协程而非线程。
- `engine_actor`（C++）：收到 `WorkloadMsg` 后按依赖跑 kernel，完成发 `WorkloadDoneMsg`。

推理侧的消息表在 `[messages.py](../src/python/hybridsim_infer/messages.py)`，`build_inference_simulation` 里统一 `register_messages(INFER_MESSAGE_TYPES)`。

平台本身的构建、C++ API、Frontier 复现见仓库根 [README.md](../README.md)。

---



## 5. 现状边界

写清楚哪些是「已经在跑的」，哪些只是框架图上的位置：


| 能力               | 现状                                                     |
| ---------------- | ------------------------------------------------------ |
| monolith / PD 分离 | 已实现                                                    |
| AF 分离等其它实例间拓扑    | 未实现                                                    |
| 集群分发策略           | 只有 least-load                                          |
| replica 内调度      | vLLM V1 语义，逐步对齐测试                                      |
| 本地 prefix cache  | 已实现（token 列表 / `hash_ids` 两种 key 来源）                   |
| 远端 KV Store      | Mooncake 风格交互骨架，非真 RDMA                                |
| batch 时长         | fixed / token-proportional / Frontier RF / op-level 解析 |
| 网络拥塞、QoS、CC      | 可选流级仿真（`network_sim`）；默认仍是 α-β。无包级 / 窗口 CC             |
| 计算 / 通信 overlap  | op DAG 依赖 + Put 立即返回；无独立 device 占用模型                   |


---



## 6. 阅读顺序

1. 本文
2. [request_generation.md](request_generation.md)：请求怎么来、`InferenceRequest` 字段
3. [scheduler.md](scheduler.md)：Cluster 分发 + replica 内 `schedule_step` + vLLM 对齐
4. [kv.md](kv.md)：本地 APC、Store、PD 控制面 lookup
5. [op_level_workload_generator.md](op_level_workload_generator.md)：batch → 算子 DAG → Roofline / α-β
6. [engine.md](engine.md)：kernel DAG 怎么被执行
7. [network.md](network.md)：可选流级网络
8. [inference_config.md](inference_config.md)：配置分组与字段
9. [examples/inference/README.md](../examples/inference/README.md)：跑起来

