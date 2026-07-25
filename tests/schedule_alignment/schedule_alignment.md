# Schedule alignment（代码视角）

本文从**代码拆解**说明：hybridsim 如何在「实例内部调度」（WAITING/RUNNING → batch）上对齐 vLLM V1 `Scheduler`，以及如何用 offline ledger 校验。对应设计意图见 [`hybridsimdesign/hybridsim inference offline校准.md`](../../../hybridsimdesign/hybridsim%20inference%20offline校准.md)。

**对齐范围（本次）**

| 关注 | 不关注 |
|------|--------|
| Replica 内 `schedule()` 语义 | Cluster → Replica 分发 |
| 每步 `scheduled_tokens` / preempt / finish | 真实 GPU / CUDA kernel |
| 本地 KV 块预算与 OOM 抢占 | Mooncake / 远程 KV 传输时序细节 |

---

## 1. 总体架构：两边各跑什么

```text
                    同一 CaseSpec（请求到达 + 调度旋钮）
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   hybridsim_schedule_driver          vllm_schedule_driver
   FrameworkFactory.create("vllm")    vllm.v1.core.sched.Scheduler
   VllmFramework.schedule_step()      Scheduler.schedule()
   VllmFramework.on_batch_complete()  update_from_output(fake ModelRunnerOutput)
              │                               │
              └──────────┬────────────────────┘
                         ▼
              ScheduleStepRecord ledger（按 step 比对）
```

| 角色 | hybridsim | vLLM（offline driver） |
|------|-----------|------------------------|
| 调度实现 | [`VllmFramework`](../../../src/python/hybridsim_infer/frameworks/vllm.py) | `vllm.v1.core.sched.scheduler.Scheduler` |
| 工厂入口 | [`FrameworkFactory`](../../../src/python/hybridsim_infer/frameworks/factory.py) | （真实库，driver 直接构造） |
| KV 容量 | [`VllmKvCacheManager`](../../../src/python/hybridsim_infer/kv_system/cache.py) | `KVCacheManager.allocate_slots` + `BlockPool` |
| 假执行 | `on_batch_complete` 推进 computed/output | `ModelRunnerOutput` 采样，不跑模型 |
| 仿真 Actor 路径 | `ReplicaSchedulerActor` → `framework.schedule_step` | 不对齐 DES，仅校准 schedule |

关键抽象：

```python
# hybridsim_infer/frameworks/base.py
class InferenceFramework(ABC):
    def schedule_step(...) -> ScheduleResult: ...
    def on_batch_complete(...) -> list[InferenceRequest]: ...  # 本步新完成的请求
```

后续挂 SGLang 等：实现子类并 `FrameworkFactory.register("sglang", ...)`，校准 harness 可复用。

---

## 2. 单步调度：与 vLLM `schedule()` 的阶段对应

vLLM V1 单次 `schedule()` 大致是：

1. **Phase 1**：遍历 `running`，在 token budget 下 chunk，失败则 FCFS preempt  
2. **若本步发生 preempt → 不再 admit waiting**  
3. **Phase 2**：遍历 `waiting`，prefix/远程 KV（可选）后 allocate 并准入  
4. 产出 `SchedulerOutput`（`num_scheduled_tokens` 等）

hybridsim 一一映射在 `VllmFramework.schedule_step`：

```text
schedule_step
  ├─ process_running_queue   # Phase 1
  ├─ preempted → waiting 队头（reversed insert）
  ├─ if not preempted: process_wait_queue  # Phase 2
  └─ build_batch → ScheduleResult
```

对应代码：[`frameworks/vllm.py`](../../../src/python/hybridsim_infer/frameworks/vllm.py) 中 `schedule_step` / `process_running_queue` / `process_wait_queue`。

Actor 侧在 `ReplicaSchedulerActor.on_step` 调用同一套；offline 校准在 `hybridsim_schedule_driver.run_hybridsim_schedule` 调用同一套——保证「仿真步进」与「ledger 驱动」语义一致。

---

## 3. 语义逐项对齐（代码级）

### 3.1 Chunked prefill / decode 配额

| 行为 | vLLM | hybridsim |
|------|------|-----------|
| Prefill 只调度剩余 prompt | `num_new_tokens` 受 prompt 与 chunk 限制 | `chunk_limit`：`min(remaining_prefill, threshold)` |
| Long prefill 阈值 | `long_prefill_token_threshold` | 同名构造参数；`0` 时回退 `tokens_per_step` |
| Decode 默认每步 1 token | 常见 path 每次 1 | `decode_tokens_per_step=1` |
| 全局 token budget | `max_num_batched_tokens` | `token_budget` / `max_num_scheduled_tokens` |
| 并发 running 上限 | `max_num_seqs` | `max_num_running_reqs` |

**注意**：chunk 只限制「本步算多少」；准入时另有 full-ISL 门控（见 3.3）。

### 3.2 Phase 1：allocate 失败 → FCFS preempt（含自抢占）

vLLM（`scheduler.py` Phase 1）：`allocate_slots` 失败则 `running.pop()`（FCFS 末尾），可抢占**当前正在尝试调度的请求本身**。

hybridsim：

```python
# VllmFramework.process_running_queue
while True:
    new_blocks = kv_cache_manager.allocate(request, num_new)
    if new_blocks is not None:
        break
    # 即使 running 只剩 1 个，也会 preempt（自抢占）
    running, token_budget, preempted = self._preempt_fcfs(...)
    if preempted is request:
        break
```

`_preempt_fcfs`：释放 KV、`num_computed_tokens=0`、状态 `PREEMPTED`，并从本步已 schedule 的 map 中撤回该请求的 token。

本步若有 preempt：**跳过 Phase 2**（与 vLLM `if not preempted_reqs: ... waiting` 一致）。

### 3.3 Phase 2：`scheduler_reserve_full_isl`

vLLM 默认 `SchedulerConfig.scheduler_reserve_full_isl=True`：WAITING/PREEMPTED 准入时，`allocate_slots(..., full_sequence_must_fit=True)` 要求**当前完整序列长度**（`request.num_tokens`）能放入空闲块，而不是只够本 chunk。

hybridsim：

```python
# process_wait_queue，reserve_full_isl=True（默认）
full_tokens = request.num_tokens  # prefill + num_output_tokens
if not kv_cache_manager.can_fit(request, full_tokens):
    break  # 停止继续 admit
# 通过后再 allocate(request, num_new) 只增长本 chunk 所需块
```

这是 `preempt_oom` case 能对齐的关键：小 KV 池下第二请求不能因「第一 chunk 只要 1 块」而被错误准入。

### 3.4 Null block

vLLM `BlockPool` 预留 null block，可用块 = `num_blocks - 1`。

hybridsim `KvCacheManager.__post_init__`：

```python
self._null_reserved = 1 if self.num_gpu_blocks > 0 else 0
self.free_blocks = max(0, self.num_gpu_blocks - self._null_reserved)
```

校准 case 里 `num_gpu_blocks` 与 vLLM `KVCacheConfig.num_blocks` 使用同一配置值。

### 3.5 KV allocate：按「computed + new」扩容

vLLM `allocate_slots`：按需要的总 token 槽位增长块，已有块可复用。

hybridsim：

```python
need_tokens = computed + num_tokens
grow = blocks_needed_to_hold(request, need_tokens)
# grow > free → None（OOM）
```

### 3.6 Prefill 完成当步采样 +1 output

vLLM：`schedule()` 已把 `num_computed_tokens` 推进到本步 scheduled 量；`update_from_output` 在 prompt 已覆盖时追加 1 个 sampled token（`num_tokens` 变长，computed 语义与 output 分离）。

hybridsim 用 `on_batch_complete` 收拢（driver 与 `ReplicaSchedulerActor.on_batch_end` 共用）：

1. `num_computed_tokens += scheduled_n`  
2. 若本步刚完成 prefill 且仍有 decode 配额：`num_output_tokens += 1`，并为 finished 检测再 `computed += 1`  
3. `num_tokens = num_prefill + num_output` 供后续 full-ISL；**preempt 不重置 `num_output_tokens`**（与 vLLM 输出仍留在 request 上一致）

### 3.7 Prefix caching

| | vLLM | hybridsim |
|--|------|-----------|
| 默认 | APC 需显式开启 | `enable_prefix_caching=False` |
| 实现 | block hash / APC | token-list 最长前缀（非 APC） |

校准 case（如 `local_prefix`）两侧默认关缓存，保证「同 prompt 也全量 prefill」一致。HS 本地前缀仅作可选仿真能力，**不声称与 vLLM APC 字节级对齐**。

### 3.8 配置旋钮对照

| Case / Config 字段 | vLLM | hybridsim |
|--------------------|------|-----------|
| `max_num_scheduled_tokens` | `max_num_batched_tokens`（且 ≥ `max_num_seqs`） | 同左；driver 会抬到 ≥ `max_num_running_reqs` |
| `max_num_running_reqs` | `max_num_seqs` | 同左 |
| `long_prefill_token_threshold` | 同名 | 同名 |
| `reserve_full_isl` | `scheduler_reserve_full_isl` | `VllmFramework.reserve_full_isl` |
| `num_gpu_blocks` / `block_size` | `KVCacheConfig` / `CacheConfig` | `KvCacheManager` |
| `enable_prefix_caching` | `CacheConfig.enable_prefix_caching` | framework 开关 |
| `framework` | — | `"vllm"`（工厂名） |

---

## 4. Offline 校准链路（如何验证对齐）

### 4.1 输入

[`cases/*.json`](cases/)：请求到达步、prefill/decode 长度、`prompt_token_ids`、scheduler 旋钮。  
[`case_loader.CaseSpec`](case_loader.py) 可带 `framework`（默认 `vllm`）。

### 4.2 hybridsim 侧

[`hybridsim_schedule_driver.py`](hybridsim_schedule_driver.py)：

1. `FrameworkFactory.create(...)`  
2. 循环：按 `arrive_step` 入 waiting → `framework.schedule_step` → 记 ledger → `on_batch_complete`  
3. 输出 `*.hybridsim.ledger.jsonl`

### 4.3 vLLM 侧

[`vllm_schedule_driver.py`](vllm_schedule_driver.py)：

1. 构造 CPU + offline HF 的 `Scheduler`（不启 Engine）  
2. `schedule()` 后用 fake `ModelRunnerOutput` 做 `update_from_output`  
3. 从 `num_scheduled_tokens` / preempt 集合等抽出同构 ledger

### 4.4 比对

[`compare.compare_ledgers`](compare.py)：默认丢掉「空闲步」，对齐比较：

- `scheduled_tokens`
- `preempted_ids`
- `finished_ids`

（可选 `compare_queues` 比 waiting/running。）

CLI：`PYTHONPATH=src/python:tests:. python -m schedule_alignment.run_case --case <name>`（默认要求 vLLM）。

测试：`tests/test_schedule_alignment.py`（每个 case 为 `subTest`：expected golden + vLLM compare）。

### 4.5 当前覆盖 case

| Case | 验证点 |
|------|--------|
| `chunked_prefill` | 长 prompt 分 chunk |
| `multi_decode` | 多请求 decode |
| `mixed_batch` | prefill+decode 混合 budget |
| `budget_exhaust` | token budget 耗尽 |
| `preempt_oom` | null block + full-ISL + 自抢占 |
| `local_prefix` | 关缓存时两侧全量 prefill |

---

## 5. 代码导读（建议阅读顺序）

1. [`frameworks/base.py`](../../src/python/hybridsim_infer/frameworks/base.py) — 扩展点  
2. [`frameworks/vllm.py`](../../src/python/hybridsim_infer/frameworks/vllm.py) — 对齐核心  
3. [`kv_system/cache.py`](../../src/python/hybridsim_infer/kv_system/cache.py) — null block / allocate / can_fit  
4. [`request.py`](../../src/python/hybridsim_infer/request.py) — `num_tokens` / `num_output_tokens`  
5. [`actors/replica_scheduler.py`](../../src/python/hybridsim_infer/actors/replica_scheduler.py) — DES 里如何调用 framework  
6. [`hybridsim_schedule_driver.py`](hybridsim_schedule_driver.py) + [`vllm_schedule_driver.py`](vllm_schedule_driver.py) — 双端 ledger  
7. 对照源码：`vllm-main/vllm/v1/core/sched/scheduler.py`、`kv_cache_manager.py`（`allocate_slots` / `full_sequence_must_fit`）

设计侧长文：[`hybridsimdesign/vLLM Engine schedule 逐行注释分析.md`](../../../../hybridsimdesign/vLLM%20Engine%20schedule%20逐行注释分析.md)。

---

## 6. 已知差距（有意不声称对齐）

- **APC / Store block hash**：本地 6 case 默认不启 Store；Mooncake 池 profile 与 vLLM 兼容 hash 见 `tests/mooncake_alignment/`  
- **Spec decode / LoRA / encoder budget / watermark>0**：未建模  
- **`num_computed` vs `num_tokens`**：HS 用 `computed`+`output` 近似；finished 条件用 `computed >= prefill+decode`  
- **Cluster 负载均衡、真实执行时长**：不在本校准范围；时长可用 `TokenProportionalPredictor` 假执行  

---

## 7. 扩展其他框架时怎么复用

1. 实现 `InferenceFramework`（`schedule_step` + `on_batch_complete`）  
2. `FrameworkFactory.register("sglang", SgLangFramework)`  
3. `InferenceConfig(framework="sglang")` 或 case JSON `"framework": "sglang"`  
4. 另写 `sglang_schedule_driver`（截获该框架 schedule 输出），ledger schema 可共用  

hybridsim 对齐的是**调度决策序列**，不是把 vLLM 嵌进仿真器；`VllmFramework` 是按 vLLM 语义手写的仿真模型，用真实 `Scheduler` offline 跑分做回归。
