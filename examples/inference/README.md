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
| KV Store / KV Client | `KvStoreActor` + `KvClient`（`enable_kv_client=True`） |
| schedule / batch | `hybridsim_infer.frameworks`（默认 `VllmFramework`；`FrameworkFactory` 可扩展） |
| Fake GPU duration | `hybridsim_infer.predictors`（`fixed` / `token_proportional`） |

## Mooncake-style KV（交互骨架）

对齐 **调度阶段**（非真 RDMA / mooncake_master）：

| 组件 | 职责 |
|------|------|
| `KvStoreActor` | Store master：block-key 元数据、容量、LRU；支持 async `KVLookupMsg` → `KVLookupReplyMsg` |
| `KvClient` / `KvClientEngine` | Replica：Store RPC + 独立传输 EngineActor 封装 |
| Store hit | 满块 `block_keys` 连续前缀 |
| Store async lookup | fire LookupMsg；Reply **只缓存**；调度 `pending`≈`None`，下步再 hit |
| Load / P2P RDMA 仿真 | allocate 后 `after_alloc_load` → TimeoutKernel；仅 **pull** TransferEnd 清 `WAIT_FOR_REMOTE_KVS` |
| P2P | `kv_mode=p2p`：固定地址 `lookup_p2p`；Prefill handoff → Decode RDMA sim；见 `pd_disagg_prefix_demo.py` |
| Save（store） | BatchEnd → `save` → `submit_push` |
| Prefill handoff + prefix | handoff 前 `cache_prefix`，同 prompt 后续请求可在 P 侧本地命中 |

配置：`enable_kv_client`、`kv_mode`（`store`\|`p2p`）、`kv_lookup_async`、`kv_lookup_rtt_s`、`kv_p2p_prefill_replica`、`kv_p2p_decode_replica`、`kv_bandwidth_gbps`、`kv_bytes_per_token`、`kv_transfer_s`。

**非目标**：真 bootstrap 端口、TransferEngine RDMA、与 vLLM 相同的 block hash、Prefill 侧 push 算子。

## Schedule alignment（测试）

调度对齐套件在 **[`tests/schedule_alignment/`](../../tests/schedule_alignment/)**（不是 example）。

文档：[schedule_alignment.md](../../tests/schedule_alignment/schedule_alignment.md) · 单元测试：`tests/test_schedule_alignment.py`

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## Run demo / tests

```bash
# Monolithic + Store seed hit
PYTHONPATH=src/python:. python examples/inference/monolithic_demo.py

# PD disagg (P→D handoff + RDMA sim) with local prefix cache
PYTHONPATH=src/python:. python examples/inference/pd_disagg_prefix_demo.py

PYTHONPATH=src/python:. python tests/test_inference_skeleton.py -v
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. \
  python tests/test_mooncake_alignment.py -v
```

`InferenceConfig(duration_mode="token_proportional")` 时 batch 时长 ∝ prefill/decode token 数。

`InferenceConfig(framework="vllm")` 选择 replica 内调度实现；扩展时实现 `InferenceFramework` 子类并 `FrameworkFactory.register("sglang", …)`。

### PD + prefix demo 要点

`pd_disagg_prefix_demo.py`：`kv_mode=p2p`、两副本、`enable_prefix_caching=True`。

1. 请求进 Prefill → 算完 handoff → Decode `lookup_p2p` + pull → decode  
2. 同 prompt 的后续请求可在 Prefill 侧命中本地 prefix cache  
3. 部分共享 prefix 的请求只命中公共前缀长度  

## Relation to Frontier

`examples/frontier` 仍是 Frontier 对齐参考；本包不依赖 Frontier。
