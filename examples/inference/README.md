# Inference simulation examples (hybridsim_infer)

Native Actor-based inference on hybridsim. Corresponds to
`hybridsimdesign/基于actor系统的推理仿真设计.md` and
`hybridsimdesign/hybridsim inference offline校准.md` (**NO_NETWORK**).

## Layout vs design doc

| Design | Package |
|--------|---------|
| ClusterActor | `hybridsim_infer.actors.ClusterActor` |
| ClusterManager | `hybridsim_infer.cluster`（`Monolith` / `Pd`） |
| ReplicaActor | `hybridsim_infer.actors.ReplicaActor`（同构，无角色） |
| WorkerEngine | `hybridsim_infer.actors.WorkerEngine` |
| KV Store / KV Client | `KvStoreActor` + `KvClient`（`enable_kv_client=True`） |
| schedule / batch | `hybridsim_infer.schedulers`（默认 `VllmScheduler`；`SchedulerFactory` 可扩展） |
| Request arrivals | `hybridsim_infer.request_generators`（`List` / ServeGen → `schedule_arrivals`） |
| Fake GPU duration | `hybridsim_infer.workload_generators`（`fixed` / `token_proportional` / `predict`） |
| Request profile | `hybridsim.request_profile`（独立进程写 Chrome Trace JSON → `profile/`） |

**RequestGenerator vs InferWorkloadGenerator**：前者生成带 `arrived_at` 的 `InferenceRequest` 序列并注入 ClusterActor；后者把已调度的 `ScheduleBatch` 变成 Engine TimeoutKernel。ServeGen 虽自称 workload generator，在本项目中只作为请求到达/长度采样后端。

请求生成专项文档（数据来源、KV 轨迹生成、数据结构）：[`docs/request_generation.md`](../../docs/request_generation.md)。

## Request profile（Chrome Trace）

`InferenceConfig(enable_request_profile=True)` 时，仿真在**子进程**收集事件并写出 JSON（默认 `<repo>/profile/request_profile.json`，目录已 gitignore）。

轨道：

| Process | Tracks |
|---------|--------|
| `Cluster` | `schedule`（`ClusterSchedule`，dur≈0）、`dispatch`（`Dispatch` → replica） |
| `Replica_N` | `engine`（`EngineReq` / `KvPull` / `KvPush`）、`schedule`（`ReplicaEnqueue` / `ReplicaSchedule`） |

Flow 箭头（Chrome Trace `ph=s/f`）：`ClusterToReplica`（Dispatch → ReplicaEnqueue）、`ScheduleToEngine`（ReplicaSchedule → EngineReq）。

请求元信息写在 `metadata.requests[<request_id>]`（arrived_at、prefill/decode token、prompt_len/prefix、kv_transfer_params、完成态等）；Dispatch / Enqueue / EngineReq 的 `args` 也带精简字段，方便 UI 悬停查看。

打开方式：Chrome `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev/) 加载 JSON。

```bash
PYTHONPATH=src/python:. python examples/inference/pd_multipool_profile_demo.py
# → profile/pd_multipool_profile_demo.json
```

CLI：`--enable_request_profile` / `--request_profile_path` / `--request_profile_dir`。

## Topology（`cluster_type`）

| 类型 | 行为 |
|------|------|
| `monolith` | 全副本 least-load；请求无 PD 盖章 |
| `pd` | Prefill 池 / Decode 池 least-load；arrive 盖 `do_remote_decode`；handoff 盖 `do_remote_prefill` + `remote_replica_id=源P` |

PD 配置：`num_prefill_replicas` / `num_decode_replicas`（总副本 = 二者之和）。Replica **不**区分身份，行为只看请求字段。

## Mooncake-style KV（交互骨架）

对齐 **调度阶段**（非真 RDMA / mooncake_master）：

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

配置：`cluster_type`、`enable_kv_client`、`model_preset`（KV 体积从 preset YAML 计算）、`kv_lookup_async`、`kv_lookup_rtt_s`、`kv_bandwidth_gbps`、`kv_transfer_s`。

**Store 正交于拓扑**：PD / Monolith 均可 `enable_kv_client=True` 挂共享 Store。

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

# 2P+2D + KV + prefix cache — best for visualizing the request profile Gantt
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

`InferenceConfig(duration_mode="batch_level", batch_predictor="token_proportional")` 时 batch 时长 ∝ prefill/decode token 数。

`InferenceConfig(framework="vllm")` 选择 replica 内调度实现；扩展时实现 `InferenceScheduler` 子类并 `SchedulerFactory.register("sglang", …)`。

### PD + prefix demo 要点

`pd_disagg_prefix_demo.py`：`cluster_type=pd`、1P+1D、`enable_prefix_caching=True`。

1. 请求进 Prefill 池 → 算完 handoff → Decode 控制面 lookup + pull → decode  
2. 同 prompt 的后续请求可在 Prefill 侧命中本地 prefix cache  
3. 部分共享 prefix 的请求只命中公共前缀长度  

## Relation to Frontier

`examples/frontier` 仍是 Frontier 对齐参考；本包不依赖 Frontier。
