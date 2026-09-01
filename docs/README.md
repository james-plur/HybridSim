# hybridsim 文档

`hybridsim_infer` 推理仿真的文档入口。先看 [architecture.md](architecture.md)（全景图 + 各层职责），再按需要深入某一层。

## 按分层阅读

| 分层 | 文档 | 讲什么 |
|------|------|--------|
| 全景 | [architecture.md](architecture.md) | 五层结构、数据流、一次请求的生命周期、现状边界 |
| 请求生成 | [request_generation.md](request_generation.md) | `InferenceRequest` 字段、List / ServeGen / KV trace 三种来源 |
| 集群 + 实例调度 | [scheduler.md](scheduler.md) | 分发拓扑、`schedule_step` 六个流程、与 vLLM V1 的对照与对齐测试 |
| KV | [kv.md](kv.md) | 本地 APC、远端 Store、PD 控制面 lookup、传输时长 |
| 计时 | [op_level_workload_generator.md](op_level_workload_generator.md) | batch → 算子 DAG → Roofline / α-β |
| Engine | [engine.md](engine.md) | kernel DAG 执行、inflight 槽位、为什么只有 TimeoutKernel |
| 配置 | [inference_config.md](inference_config.md) | 嵌套 `InferenceConfig` 的七个分组与字段 |

## 上手

跑 demo、看 Chrome Trace、跑测试：[examples/inference/README.md](../examples/inference/README.md)。

平台本身（C++ Actor 系统、构建、Frontier 复现）：仓库根 [README.md](../README.md)。

## 相关测试文档

| 目录 | 校准对象 |
|------|----------|
| `tests/schedule_alignment/` | 与真实 vLLM `Scheduler` 的逐步决策对齐 |
| `tests/mooncake_alignment/` | Store / KV 交互时序 |
| `tests/analytic_workload_calibration/` | op-level 解析时长的离线校准 |
