# Mooncake 对齐（调度 + Store 池 profile）

本文从**代码角度**说明：hybridsim 如何与 **vLLM + Mooncake Store / PD** 对齐，以及如何跑 case。设计意图与测试方法论见  
[`hybridsimdesign/hybridsim inference offline校准.md`](../../../hybridsimdesign/hybridsim%20inference%20offline校准.md)。

本目录是**第二阶段** offline 校准（在本地 schedule ledger 之上）：

1. **调度（Schedule）** — 与 [`../schedule_alignment/`](../schedule_alignment/) 共用 `ScheduleStepRecord`
2. **Mooncake Store 池** — CRUD 事件 JSONL（`exist` / `put` / `get` / `evict`），含 `step`、`hashes`、可选 `keys`

PD（`MooncakeConnector`）用 DES `cluster_type=pd` 做 handoff smoke；**池 profile 仅针对 Store**。

---

## 1. 对齐对象与代码映射

```text
  CaseSpec
     │
     ├─► vLLM + MooncakeStoreConnector (+ pool_profile hook)
     │         │                    │
     │         ▼                    ▼
     │   schedule ledger      *.vllm.pool.jsonl
     │
     └─► hybridsim（offline driver 或 DES）
               │                    │
               ▼                    ▼
         schedule ledger      *.hybridsim.pool.jsonl
               │                    │
               └────────┬───────────┘
                        ▼
              compare_ledgers / compare_pool_profiles
```

| 关注点 | 真实侧（vLLM + Mooncake） | 仿真侧（hybridsim） |
|--------|---------------------------|---------------------|
| 实例内调度 | `vllm.v1.core.sched.Scheduler` | `frameworks.VllmFramework.schedule_step` |
| 本地 GPU 块 | `KVCacheManager` / BlockPool（ref_cnt、命中复用、allocate 时挂满块） | `kv_system.VllmKvCacheManager`（轻量 BlockPool；`num_gpu_blocks<=0` 无限） |
| Store 元数据池 | `mooncake_master` + Store connector（DRAM；压力可落盘） | `MooncakeKvStore`：DRAM LRU；`kv_store_blocks<=0` 无限 DRAM |
| 写池 / put 量 | 满块门控 + 增量 suffix put | `save_computed_prefixes`：`num_saved` + `aligned` 门控；`insert_keys` 返回增量 token |
| Store worker CRUD 打点 | `mooncake/store/pool_profile.py` + `worker.py` | `_emit` / `pool_recorder`（`exist` / `put` / `get` / `evict`） |
| Store 客户端 | connector / Mooncake client | `kv_system.KvClient` |
| 传输时长仿真 | 真实 RDMA（本阶段不对齐数值） | DRAM：`α_net+B/BW_net`（`kv_bandwidth_gbps` / `kv_latency_s`） |
| Store Actor 外壳 | （进程外 master） | `actors.KvStoreActor`（只收 Msg） |
| PD handoff | `MooncakeConnector` + proxy | Cluster `RequestHandoffMsg` + `cluster_type=pd` |
| Block hash | vLLM `hash_block_tokens` | `kv_system.block_keys`（同算法） |

**关键约束**：offline [`hybridsim_store_driver`](hybridsim_store_driver.py) 与 DES `KvStoreActor` **调用同一套** `MooncakeKvStore.lookup_keys` / `insert_keys` / `get_keys`，避免测试替身与仿真分叉。

Replica 远端决策统一走 Manager：

```text
ReplicaSchedulerActor
  └─ VllmKvCacheManager
        ├─ 本地 match / allocate / free / cache_prefix
        └─ attach_client(KvClient)
              ├─ Store RPC → KvStoreActor → MooncakeKvStore
              └─ pull/push → KvClientEngine（平台 EngineActor）
```

---

## 2. 两条轨道如何对齐

### 轨道 A：Schedule

- Schema：`schedule_alignment.schema.ScheduleStepRecord`
- 比对：`schedule_alignment.compare.compare_ledgers`
- Store case：`remote_lookup` → `MooncakeKvStore.lookup_keys`；hit 后 `WAIT_FOR_REMOTE_KVS` + `get`，再减少剩余 prefill 的 `scheduled_tokens`
- PD case：DES Prefill handoff → Decode control-plane lookup（跳过 Store hash）+ RDMA 仿真；handoff 前可 `cache_prefix`（本地 prefix）

### 轨道 B：Store 池 profile

每行 JSONL 为 [`MooncakePoolEvent`](schema.py)：

| 字段 | 含义 |
|------|------|
| `step` | schedule 步（或 worker 单调序号） |
| `op` | `exist` \| `put` \| `get` \| `delete` \| `evict` |
| `hashes` | block hash hex（可只比 PoolKey 后缀） |
| `keys` | 完整 store key（可选） |
| `hit_mask` / `num_tokens` | exist/get 摘要 |
| `req_id` | 可选 |

比对：[`compare_pool_profiles`](compare.py) 按 `(step, op, 排序后的 hashes, num_tokens)`。忽略绝对 key 前缀差异。

**真实侧挂钩**（`vllm-main`）：

- `exist` / `put`：`KVCacheStoreSendingThread`（`batch_is_exist` → `batch_put`）
- `get`：`KVCacheStoreRecvingThread`
- lookup：`MooncakeStoreWorker.lookup`
- 开关：`VLLM_MOONCAKE_POOL_PROFILE=1`，路径 `VLLM_MOONCAKE_POOL_PROFILE_PATH`

**仿真侧**：

- offline：`MooncakeKvStore` + `pool_recorder`
- DES：`KvStoreActor.store` 同为 `MooncakeKvStore`，`profile_fn` 可选

Hash：`PYTHONHASHSEED=0` + `block_keys.hash_block_tokens`（对齐 vLLM pickle SHA-256 链）。

---

## 3. 环境准备

```bash
export PYTHONHASHSEED=0   # block hash 可复现，必须
uv pip install mooncake-transfer-engine   # 或自编译 Mooncake
# 可选：真实 vLLM Store 跑分
```

检查 / 启动 master：

```bash
bash tests/mooncake_alignment/scripts/check_env.sh
bash tests/mooncake_alignment/scripts/start_master.sh
export MOONCAKE_CONFIG_PATH=$PWD/tests/mooncake_alignment/scripts/mooncake_config.tcp.json
```

实验室无 RDMA 时用 TCP 配置；有 RDMA 时把 `protocol` 改为 `rdma`。

---

## 4. 如何跑 hybridsim offline Store case

```bash
cd hybridsim
PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. \
  python -c "
from pathlib import Path
import json
from schedule_alignment.case_loader import CaseSpec
from mooncake_alignment.hybridsim_store_driver import run_hybridsim_store_case
case = CaseSpec.from_dict(
    json.load(open('tests/mooncake_alignment/cases/store_prefix_hit.json')),
    name='store_prefix_hit')
run_hybridsim_store_case(
    case,
    schedule_out=Path('tests/mooncake_alignment/_out/store_prefix_hit.hybridsim.ledger.jsonl'),
    pool_out=Path('tests/mooncake_alignment/_out/store_prefix_hit.hybridsim.pool.jsonl'))
print('ok')
"
```

Case JSON 可用 `scheduler.store_num_blocks` 限制 offline DRAM 容量（默认 4096；`<=0` 无限），用于触发 LRU `evict`。

Offline put 与 DES 一致：仅当 `floor(computed/bs)*bs` 越过已 save 满块边界时，对增量 keys 调用 `insert_keys`。

---

## 5. vLLM Store 池 hook

```bash
export VLLM_MOONCAKE_POOL_PROFILE=1
export VLLM_MOONCAKE_POOL_PROFILE_PATH=/tmp/run.vllm.pool.jsonl
export PYTHONHASHSEED=0
# 再启动：kv_connector=MooncakeStoreConnector 的 vLLM
```

实现：`vllm/.../mooncake/store/pool_profile.py`，在 `worker.py` 的 exist/put/get 路径调用 `record`。

---

## 6. 测试

```bash
PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. python tests/test_mooncake_alignment.py -v
PYTHONPATH=src/python:. python tests/test_inference_skeleton.py -v
# PD + 本地 prefix demo
PYTHONPATH=src/python:. python examples/inference/pd_disagg_prefix_demo.py
```

无 `mooncake_master` / 真 RDMA 时**跳过真实侧**；hybridsim 自洽与语义断言始终执行。

---

## 7. Case 一览

| case | 轨道 |
|------|------|
| `store_prefix_hit` | 全量 remote hit（两请求同 prompt） |
| `store_prefix_partial_hit` | **部分命中**：共享 2/3 block，后缀继续 prefill |
| `store_prefix_nested_partial` | 嵌套前缀：后续分别 hit 16 / 8 tokens |
| `store_prefix_evict` | **LRU evict**：`store_num_blocks=2`，挤掉后同 prompt miss |
| `pd_handoff_decode` | DES P2P handoff smoke（无 pool profile） |

---

## 8. 本阶段非目标

- 真 RDMA 带宽/时延数值对齐  
- P2P TransferEngine 字节级 profile  
- 改写阶段一 6 个纯本地 schedule golden（本地 case 不启 Store）  
- 每个 step 整池快照（可用 put−evict 事件重放重建；比对以 CRUD 轨迹为准）
