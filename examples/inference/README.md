# Inference simulation examples (hybridsim_infer)

Native Actor-based inference on hybridsim. Corresponds to
`hybridsimdesign/基于actor系统的推理仿真设计.md` and
`hybridsimdesign/hybridsim inference offline校准.md` (**NO_NETWORK**).

架构总览（各层职责、数据流、现状边界）：**[`docs/architecture.md`](../../docs/architecture.md)**；配置：**[`docs/inference_config.md`](../../docs/inference_config.md)**；输出：**[`docs/outputs.md`](../../docs/outputs.md)**；平台基座：**[`docs/platform.md`](../../docs/platform.md)**。仓库总览见根目录 [`README.md`](../../README.md)。

## 分层与代码

| 分层 | 代码 | 文档 |
|------|------|------|
| 请求生成 | `hybridsim_infer.request_generators`（List / ServeGen / KV trace → `schedule_arrivals`） | [request_generation.md](../../docs/request_generation.md) |
| 集群分发 | `ClusterActor` + `hybridsim_infer.cluster`（`Monolith` / `Pd`） | [scheduler.md](../../docs/scheduler.md) |
| 实例调度 | `ReplicaActor`（同构，无角色）+ `hybridsim_infer.schedulers`（默认 `VllmScheduler`） | [scheduler.md](../../docs/scheduler.md) |
| KV | `VllmKvCacheManager`、`KvClient`、`KvStoreActor`（`kv.enable_store=True`） | [kv.md](../../docs/kv.md) |
| 计时 | `hybridsim_infer.workload_generators`（`batch_level` / `op_level`） | [workload_generator.md](../../docs/workload_generator.md) |
| Engine 执行 | `WorkerEngine` + 平台 `EngineActor`（`TimeoutKernel` DAG） | [engine.md](../../docs/engine.md) |
| 观测 | `hybridsim.request_profile`（独立进程写 Chrome Trace JSON → `profile/`） | 见下节 |

**RequestGenerator vs InferWorkloadGenerator**：前者生成带 `arrived_at` 的 `InferenceRequest` 序列并注入 ClusterActor；后者把已调度的 `ScheduleBatch` 变成 Engine TimeoutKernel。ServeGen 虽自称 workload generator，在本项目中只作为请求到达/长度采样后端。

配置分组（cluster / schedule / kv / model / workload / output）：[`docs/inference_config.md`](../../docs/inference_config.md)；输出与 trace：[`docs/outputs.md`](../../docs/outputs.md)。

## Request profile（Chrome Trace）

`InferenceConfig(output=OutputConfig(request_profile=RequestProfileOutput(enabled=True)))` 时，仿真在**子进程**收集事件并写出 JSON（默认 `<repo>/profile/request_profile.json`，目录已 gitignore）。

嵌套字段说明见 [`docs/inference_config.md`](../../docs/inference_config.md)。

轨道：

| Process | Tracks |
|---------|--------|
| `Cluster` | `schedule`（`ClusterSchedule` / `RequestFinish`）、`dispatch`（`Dispatch` / `Handoff`） |
| `Replica_N (Prefill\|Decode)` | `engine`（`EngineReq` / `KvPull` / `KvPush`）、`schedule`（`ReplicaEnqueue` / `ReplicaSchedule`） |
| 同上（`infer_workload.mode=op_level`） | `compute_*` / `comm_*`：按依赖最早开工的 kernel slice；跨 stream 依赖为 `KernelDep` flow |

Flow 箭头（Chrome Trace `ph=s/f`）：`ClusterToReplica`（Dispatch → ReplicaEnqueue）、`ScheduleToEngine`（ReplicaSchedule → EngineReq）、`KernelDep`（跨 compute/comm stream 的算子依赖）。

请求元信息写在 `metadata.requests[<request_id>]`（arrived_at、prefill/decode token、prompt_len/prefix、kv_transfer_params、完成态等）；`metadata` 还可含 demo 写入的 `input` / `workload` / `tp` / `n_requests`。Dispatch / Enqueue / EngineReq 的 `args` 带 `phase`、`scheduled_tokens`、`n_kernels`、`critical_path_s` 等，方便 UI 悬停查看。

打开方式：Chrome `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev/) 加载 JSON。看算子依赖线时优先 `chrome://tracing`，选中 slice 后显示 flow。

```bash
PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py
# → profile/pd_multipool_profile_demo.json  （handwritten + batch_level）

PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py --input trace
# 真实 KV trace 前 10 条（默认 mooncake_fast25）

PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py --workload op
# llama-3.1-8b 全层 + TP=2；profile 含 compute/comm stream

PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py \
  --input trace --workload op --max-requests 10
```

CLI：`--input handwritten|trace`、`--workload batch|op`、`--trace`、`--max-requests`、`--max-decode`、`--tp`、`--profile-path`、`--write-metrics`。脚本默认打印 `metrics()`。

## Topology（`cluster.type`）

| 类型 | 行为 |
|------|------|
| `monolith` | 全副本 least-load；请求无 PD 盖章 |
| `pd` | Prefill 池 / Decode 池 least-load；arrive 盖 `do_remote_decode`；handoff 盖 `do_remote_prefill` + `remote_replica_id=源P` |

PD 配置：`ClusterConfig(type="pd", num_prefill_replicas=..., num_decode_replicas=...)`（总副本 = 二者之和）。Replica **不**区分身份，行为只看请求字段。

## Mooncake-style KV（交互骨架）

对齐 **调度阶段**（非真 RDMA / mooncake_master）；Prefix cache / Store 与 PD 传输的完整说明见 [`docs/kv.md`](../../docs/kv.md)：

| 组件 | 职责 |
|------|------|
| `KvStoreActor` | Store master：block-key 元数据、容量、LRU；支持 async `KVLookupMsg` → `KVLookupReplyMsg` |
| `KvClient` | Replica：Store RPC + 独立传输 EngineActor；PD Decode 走 control-plane lookup |
| Store hit | 满块 `block_keys` 连续前缀（Monolith / 无 PD 标志时） |
| Store async lookup | fire LookupMsg；Reply **只缓存**；调度 `pending`≈`None`，下步再 hit |
| PD Decode lookup | **跳过** Store hash 匹配；`lookup_control_plane` 仿真向源 P 的控制面 RTT，再 `after_alloc_load` pull |
| Load RDMA 仿真 | allocate 后 pull TimeoutKernel；仅 **pull** TransferEnd 清 `WAIT_FOR_REMOTE_KVS` |
| Save（若启用 Store） | BatchEnd → `save` → `submit_push`（PD Prefill 与 Monolith 均可） |
| Prefill handoff + prefix | handoff 前 `cache_prefix`，同 prompt 后续请求可在 P 池本地命中 |

配置：`cluster.type`、`kv.enable_store`、`model.preset`（KV 体积从 preset YAML 计算）、`kv.lookup.async_`、`kv.lookup.rtt_s`、`kv_workload.bandwidth_gbps`、`kv_workload.transfer_s_floor`。

**Store 正交于拓扑**：PD / Monolith 均可 `kv.enable_store=True` 挂共享 Store。

**非目标**：真 bootstrap 端口、TransferEngine RDMA、与 vLLM 相同的 block hash、Prefill 侧 push 算子。

## Schedule alignment（测试）

调度对齐套件在 **[`tests/schedule_alignment/`](../../tests/schedule_alignment/)**（不是 example）。

文档：[docs/scheduler.md](../../docs/scheduler.md)（实现 + 对齐测试）· [schedule_alignment.md](../../tests/schedule_alignment/schedule_alignment.md)（代码映射）· 单元测试：`tests/test_schedule_alignment.py`

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## Run demo / tests

```bash
# Monolithic + Store seed hit (+ request profile)
PYTHONPATH=src/python:. python examples/inference/monolithic_demo.py

# PD disagg (P→D handoff + control-plane RTT + RDMA sim) with local prefix cache
PYTHONPATH=src/python:. python examples/inference/pd_disagg_prefix_demo.py

# 2P+2D + KV + prefix cache — request profile Gantt
#   --input trace --workload op  见上文 Request profile CLI
PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py

# ServeGen RequestGenerator (optional: install ServeGen first)
#   git clone https://github.com/alibaba/ServeGen.git && pip install -e ./ServeGen
#   # or from hybridsim: pip install -e ".[servegen]"
# ClientPool reads data/ relative to the ServeGen clone; hybridsim chdirs there automatically.
PYTHONPATH=src/python:. python examples/inference/servegen_demo.py

PYTHONPATH=src/python:. python tests/test_inference_skeleton.py -v
PYTHONPATH=src/python:. python tests/test_request_generator.py -v
PYTHONPATH=src/python:. python tests/test_request_profile.py -v
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. \
  python tests/test_mooncake_alignment.py -v
```

`InferenceConfig(infer_workload=InferWorkloadConfig(mode="batch_level", batch=BatchLevelConfig(predictor="token_proportional")))` 时 batch 时长 ∝ prefill/decode token 数。

`InferenceConfig(schedule=ScheduleConfig(replica=ReplicaScheduleConfig(name="vllm")))` 选择 replica 内调度实现；扩展时实现 `InferenceScheduler` 子类并 `SchedulerFactory.register("sglang", …)`。

### PD + prefix demo 要点

`pd_disagg_prefix_demo.py`：`cluster.type=pd`、1P+1D、`kv.enable_prefix_caching=True`。

1. 请求进 Prefill 池 → 算完 handoff → Decode 控制面 lookup + pull → decode  
2. 同 prompt 的后续请求可在 Prefill 侧命中本地 prefix cache  
3. 部分共享 prefix 的请求只命中公共前缀长度  

## Relation to Frontier

`examples/frontier` 仍是 Frontier 对齐参考；本包不依赖 Frontier。
