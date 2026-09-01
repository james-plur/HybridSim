# hybridsim Scheduler 调度实现

> **本层位置**：集群分发（cluster）与 replica 内部第一步（schedule）。KV 细节见 [`kv.md`](kv.md)，batch 之后的计时与执行见 [`engine.md`](engine.md)，全景见 [`architecture.md`](architecture.md)。

本文说明 hybridsim 如何把一条请求变成一步可执行的 `ScheduleBatch`，并对照 vLLM V1 `Scheduler`。  
KV cache 的块哈希、Store / Mooncake 传输时序另文；这里只把 KV **当成调度门控**（`allocate` / `can_fit` 成功或失败），不展开其内部实现。

相关代码：

| 角色 | 路径 |
|------|------|
| 集群入口 | `hybridsim_infer/actors/cluster.py`（`ClusterActor`） |
| 拓扑派发 | `hybridsim_infer/cluster/`（`MonolithClusterManager` / `PdClusterManager`） |
| Replica 循环 | `hybridsim_infer/actors/replica.py`（`ReplicaActor`） |
| 调度策略 | `hybridsim_infer/schedulers/vllm_schedule.py`（`VllmScheduler`） |
| 调度接口 | `hybridsim_infer/schedulers/factory.py`（`InferenceScheduler` / `SchedulerFactory`） |
| Batch DTO | `hybridsim_infer/schedule_types.py` |
| 对齐测试 | `tests/schedule_alignment/`、`tests/test_schedule_alignment.py` |

请求如何生成、字段含义见 [`request_generation.md`](request_generation.md)。  
代码级逐项对照还可看 [`tests/schedule_alignment/schedule_alignment.md`](../tests/schedule_alignment/schedule_alignment.md)。  
集群分发（流程 A）的拓扑与角色约定另见 [`architecture.md`](architecture.md) 的「集群分发」一节。

---

## 1. 定位：从请求到 batch

hybridsim 把「调度」拆成两层，对应真实 serving 里 **集群入口** 和 **单实例 Engine**：

```text
RequestGenerator
      │  list[InferenceRequest]（arrived_at / prefill / decode）
      ▼
ClusterActor.schedule_arrivals          ← 集群入口：何时、派到哪台 replica
      │  RequestMsg
      ▼
ReplicaActor  waiting / running         ← 实例内部：本步算哪些请求、各算多少 token
      │  VllmScheduler.schedule_step
      ▼
ScheduleBatch                           ← 本步决策结果（chunks + tokens_per_request）
      │  InferWorkloadGenerator         ← 不在本文范围：把 batch 变成假执行时长
      ▼
WorkerEngine → BatchEndMsg
      │  VllmScheduler.on_batch_complete
      ▼
推进 computed / output；finish 或 PD handoff
```

**本文范围**是上图里「请求进入 replica 队列 → 产出 `ScheduleBatch` → 假执行后更新请求状态」。  
**不在本文范围**：Store lookup / RDMA pull、APC 哈希细节、analytical / Frontier 计时。

与 vLLM 的对应关系：

| hybridsim | vLLM V1 | 说明 |
|-----------|---------|------|
| `ClusterActor` + `ClusterManager` | 无直接对应 | vLLM 一个 `Scheduler` 管一台 Engine；多实例由外部 LB / 自研 Conductor 完成。hybridsim 把这一层显式做成 Actor。 |
| `ReplicaActor` 的 waiting / running | `Scheduler.waiting` / `Scheduler.running` | 实例内部两队列。 |
| `VllmScheduler.schedule_step` | `Scheduler.schedule()` | 一步调度决策。 |
| `ScheduleBatch` | `SchedulerOutput`（`num_scheduled_tokens` 等） | 本步要 forward 的 token 图。 |
| `on_batch_complete` | `update_from_output(ModelRunnerOutput)` | forward 之后推进 computed / 采样 output。 |
| `InferWorkloadGenerator` + `WorkerEngine` | GPU `ModelRunner` | 仿真用 TimeoutKernel 代替真实 kernel；对齐测试连这一层都不跑。 |

`VllmScheduler` **不是**把 vLLM 嵌进仿真器，而是按 vLLM 语义手写的决策模型。对齐测试用真实 `vllm.v1.core.sched.scheduler.Scheduler` 离线跑同一组 case，比对决策序列。

---

## 2. 主要流程

下面按一次请求的生命周期拆成六个流程。每个流程先写 hybridsim 做什么，再写与 vLLM 的异同。

### 流程 A：集群到达与分发

**hybridsim**

1. `schedule_arrivals` 按 `req.arrived_at` 发 `RequestArriveMsg`（DES 绝对时间，秒）。
2. `ClusterActor.on_request_arrive` 调 `ClusterManager.on_arrive(req)`，得到 `replica_id`。
3. 向该 replica 发 `RequestMsg`。

两种拓扑：

| `cluster.type` | 选谁 | 盖章 |
|----------------|------|------|
| `monolith` | 全部 replica 里 least-load | 清掉 PD 标志 |
| `pd` | Prefill 池 least-load | `do_remote_decode=True`；handoff 时再选 Decode 池 |

PD 下 Prefill 算完 prompt 后，`ReplicaActor._maybe_handoff_prefill` 发 `RequestHandoffMsg`；`PdClusterManager.on_handoff` 选 Decode replica，并把 `num_computed_tokens` 清零（Decode 侧再拉 KV）。Replica **不按角色分代码路径**，只看请求字段。

**对照 vLLM**

vLLM Engine 内部没有这一层：请求已经在某个进程的 `Scheduler.add_request`。多卡 / 多实例调度是集群产品（Mooncake Conductor、网关）的事，不在 `Scheduler.schedule()` 里。

对齐测试 **故意不覆盖** 本流程：harness 假定请求已经在「一台 replica」上。

---

### 流程 B：Replica 入队与步进循环

**hybridsim**

`ReplicaActor.on_request`：

- `req.status = WAITING`
- 追加到 `self.waiting`
- `_arm_step()` 发 `StepMsg`

`on_step` 每拍最多调一次 `schedule_step`：

1. 若 Worker 已满（`max_inflight_batches`，默认 1），本拍不调度。
2. 正在 Worker 上执行的请求从 waiting/running 里拆出去，避免同一请求被编进两个 in-flight batch。
3. `await scheduler.schedule_step(...)`，写回 waiting / running。
4. 若有 `ScheduleBatch`，`InferWorkloadGenerator` 生成 kernel → `WorkerEngine.submit`。
5. 若还有活、且 Worker 还能接，延迟 `step_interval` 再发 `StepMsg`。

`BatchEndMsg` 到来后走流程 F（`on_batch_complete`），再 `_arm_step()`。

**对照 vLLM**

| | hybridsim | vLLM |
|--|-----------|------|
| 入队 | `RequestMsg` → `waiting` | `Scheduler.add_request` → `waiting` |
| 何时 schedule | DES `StepMsg`；受 Worker inflight 门控 | Engine 循环：上一 batch 的 output 回来后再 `schedule()` |
| 并发 batch | `max_inflight_batches`（默认 1） | 通常一步一 batch；async scheduling 是另一条路径，本仿真默认关 |

语义对齐点是 **队列 + `schedule()` 决策**，不是 DES 时钟或 inflight 深度。对齐 harness 用「逻辑 step 下标」代替 wall-clock，一步 schedule、一步假完成，等价于 `max_inflight=1`。

---

### 流程 C：单步调度骨架（Phase 1 → Phase 2）

这是 `VllmScheduler.schedule_step` 的主干，直接映射 vLLM `Scheduler.schedule()`：

```text
schedule_step(waiting, running, token_budget, max_num_running_reqs)
  │
  ├─ Phase 1  process_running_queue     已在跑的请求：chunk / decode，失败则 FCFS preempt
  ├─ 被抢占的请求 reversed insert 到 waiting 队头
  ├─ 若本步发生过 preempt → 跳过 Phase 2
  └─ Phase 2  process_wait_queue        准入 WAITING：门控通过后 allocate 本 chunk，进入 running
       └─ build_batch → ScheduleResult
```

对应代码：`vllm_schedule.py` 的 `schedule_step` / `process_running_queue` / `process_wait_queue` / `build_batch`。

**对照 vLLM**

vLLM V1 同结构：

1. 遍历 `running`，在 token budget 下给新 token，`allocate_slots` 失败则 FCFS preempt。
2. `if not preempted_reqs:` 才遍历 `waiting`。
3. 产出 `SchedulerOutput`。

两边共同的硬规则：**本步一旦抢占，不再 admit 新请求**。这避免「刚腾出的块立刻被 waiting 填满、running 继续 OOM」的抖动。

---

### 流程 D：本步 token 配额（chunked prefill / decode）

在 Phase 1 / Phase 2 里，每条请求先算 `num_new`，再和剩余 `token_budget` 取 min。

`VllmScheduler.chunk_limit`：

| 阶段 | hybridsim | vLLM |
|------|-----------|------|
| Prefill | `min(剩余 prompt, long_prefill_token_threshold 或 tokens_per_step)` | `num_new_tokens` 受 prompt 剩余与 long-prefill 阈值限制 |
| Decode | 默认 `decode_tokens_per_step=1` | 常见 path 每步 1 token |
| 全局 budget | `max_num_scheduled_tokens`（Replica 传入 `token_budget`） | `max_num_batched_tokens` |
| 并发上限 | `max_num_running_reqs` | `max_num_seqs` |

`chunk_limit` 只决定「本步算多少」。WAITING 准入还有另一道门：`reserve_full_isl`（流程 E），看的是**整段当前序列**能不能放进 KV 池，而不是本 chunk。

---

### 流程 E：RUNNING 续跑、OOM 抢占、WAITING 准入

#### E.1 Phase 1：allocate 失败 → FCFS preempt（含自抢占）

对 `running` 从头扫：

1. `num_new = min(chunk_limit, token_budget)`。
2. `kv_cache_manager.allocate(request, num_new)`。
3. 失败则 `_preempt_fcfs`：弹出 **running 队尾**（最新进入 running 的），释放其 KV，`status=PREEMPTED`，并从本步已 schedule 的 map 里撤回它的 token。
4. 即使 running 只剩当前这条，也会抢占它自己（自抢占），然后结束 Phase 1。

被抢占的请求插回 waiting **队头**（`reversed(preempted)`），下一拍优先再试。`num_output_tokens` **不**因抢占清零，与 vLLM「输出仍留在 request 上」一致；`num_computed_tokens` 在 manager.preempt 里回到 0，下次要重新 prefill。

**对照 vLLM**：`scheduler.py` Phase 1 同样 `running.pop()` 队尾，允许抢占正在尝试调度的那条。hybridsim 这里是刻意对齐，不是简化。

#### E.2 Phase 2：WAITING 准入

仅当本步 **没有** preempt 时执行。对 waiting 从头扫，任一门控失败则 **停止继续 admit**（后面的请求本步全部留下 waiting）：

1. `len(running) >= max_num_running_reqs` 或 `token_budget <= 0` → 停。
2. `WAIT_FOR_REMOTE_KVS` → 跳过该条（远程 KV 未到；细节不在本文）。
3. `PREEMPTED` → 改回 `WAITING` 再走后续。
4. （可选）本地 prefix 命中：抬高 `num_computed_tokens`，从而减少后面的 `chunk_limit`。对齐测试在开 APC 的 case 里会核对这一点；哈希实现见 KV 文档。
5. （可选）`remote_lookup`：命中则排队 pull 并 `stop_after_remote`。对齐测试 **关闭** 这条路径（`remote_lookup=None`）。
6. 若命中后已经 `is_finished()`：直接 finish + free（cached-finish）。
7. **`reserve_full_isl=True`（默认）**：`can_fit(request, request.num_tokens)`，其中 `num_tokens = prefill + 已输出`。不够则停。
8. `allocate(request, num_new)` 只增长本 chunk 所需块；失败则停。
9. `status=RUNNING`，加入 running，记入本步 `scheduled_tokens`。

**对照 vLLM**：`SchedulerConfig.scheduler_reserve_full_isl=True` 时，`allocate_slots(..., full_sequence_must_fit=True)` 要求当前完整序列能放入空闲块。这是 `preempt_oom` case 能对齐的关键：小 KV 池下，不能因为「第一 chunk 只要 1 块」就把第二条请求放进来。

KV 容量在调度里只表现为：

- 可用块 = `num_gpu_blocks - 1`（预留 null block，与 vLLM `BlockPool` 一致）。
- `allocate` 按「已 computed + 本步 new」扩容；不够返回 `None`。

---

### 流程 F：组 batch、假执行、状态推进

**组 batch**（`build_batch`）

把 `scheduled_tokens` 拆成 `PrefillChunk` / `DecodeChunk`：

- `num_computed_tokens < num_prefill_tokens` → 本步先补 prefill；若 `n` 超过剩余 prompt，余量记为 decode。
- 否则整段是 decode。

得到 `ScheduleBatch(batch_id, chunks, requests, tokens_per_request, req_to_new_blocks)`。

**对照 vLLM**：`SchedulerOutput.num_scheduled_tokens` 是同一张「req → 本步 token 数」表。hybridsim 额外把 prefill/decode 拆成 chunk，方便 InferWorkloadGenerator 计时。

**假执行（仿真路径，对齐测试跳过时长）**

`ReplicaActor` 把 batch 交给 `InferWorkloadGenerator` → `WorkerEngine`。时长与调度决策正交。对齐 harness 不启 Engine，schedule 完立刻 `on_batch_complete`。

**状态推进**（`on_batch_complete` ↔ vLLM `update_from_output`）

对 batch 内每条请求：

1. `num_computed_tokens += scheduled_n`。
2. 若本步 **刚完成 prefill** 且还有 decode 配额：`num_output_tokens += 1`，并为 finished 检测再 `computed += 1`（对应 vLLM：prompt 覆盖后当步采样 1 个 output token）。
3. 若本步本来就是 decode：`num_output_tokens += n`。
4. `is_finished()`（`computed >= prefill + decode`）→ `FINISHED`，free KV，从 running 去掉；仿真路径再向 Cluster 发 `RequestFinishMsg`（或 PD handoff）。

vLLM 的差异主要在记账方式：`schedule()` 里已经把 `num_computed_tokens` 加上本步 scheduled 量，`update_from_output` 再追加 sampled token。hybridsim 把两步收进 `on_batch_complete`，对齐测试按「一步 schedule + 一步 complete」对齐最终 `scheduled_tokens` / `finished_ids`，而不是对齐 vLLM 内部两个字段的中间态。

---

### 配置旋钮对照

| hybridsim | vLLM | 默认 / 备注 |
|-----------|------|-------------|
| `max_num_scheduled_tokens` | `max_num_batched_tokens` | driver 会抬到 ≥ `max_num_running_reqs`（vLLM 约束） |
| `max_num_running_reqs` | `max_num_seqs` | |
| `tokens_per_step` / `long_prefill_token_threshold` | `long_prefill_token_threshold` | HS 阈值 `0` 时回退 `tokens_per_step` |
| `decode_tokens_per_step` | 每步 1 decode token | 默认 1 |
| `reserve_full_isl` | `scheduler_reserve_full_isl` | 默认 True |
| `num_gpu_blocks` / `block_size` | `KVCacheConfig.num_blocks` / `CacheConfig.block_size` | `num_gpu_blocks<=0` 时 HS 视为无限 |
| `enable_prefix_caching` | `CacheConfig.enable_prefix_caching` | 默认 False |
| `scheduler_name` / `SchedulerFactory` | （真实 `Scheduler` 类） | 目前只注册 `"vllm"` |

---

## 3. `schedule_alignment` 对齐测试怎么做

目标：**同一组请求、同一组旋钮下，hybridsim 的逐步调度决策与真实 vLLM `Scheduler` 一致。**  
不跑 DES、不跑 GPU、不启 Cluster、不启 Store。

### 3.1 测什么、不测什么

| 测 | 不测 |
|----|------|
| Replica 内 waiting/running → 每步 `scheduled_tokens` | Cluster 分发、least-load、PD handoff |
| preempt / finish 集合 | 真实 CUDA / ModelRunner |
| token budget、chunked prefill、full-ISL 准入 | Mooncake / 远程 KV 时序 |
| 开 APC 时的 `prefix_hit_tokens` / `free_blocks` / `allocated_blocks` | 公开 remapped `hash_ids` 与 vLLM hasher 混比 |
| null block 预留 | spec decode / LoRA / encoder budget / watermark>0 |

设计意图见 `hybridsimdesign/hybridsim inference offline校准.md`。

### 3.2 总体结构

```text
                 同一 CaseSpec（JSON fixture）
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
hybridsim_schedule_driver          vllm_schedule_driver
SchedulerFactory.create("vllm")    vllm.v1.core.sched.Scheduler
VllmScheduler.schedule_step()      Scheduler.schedule()
on_batch_complete()                update_from_output(fake ModelRunnerOutput)
           │                               │
           └──────────┬────────────────────┘
                      ▼
           ScheduleStepRecord ledger（JSONL，按逻辑 step）
                      │
              compare_ledgers
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  cases/*.expected.ledger.jsonl   真实 vLLM 输出
  （hybridsim golden 回归）        （跨实现对齐）
```

目录：

| 文件 | 作用 |
|------|------|
| `tests/schedule_alignment/cases/*.json` | 共享输入 |
| `cases/*.expected.ledger.jsonl` | 已提交的 hybridsim golden |
| `case_loader.py` | 解析 `CaseSpec` |
| `schema.py` | `ScheduleStepRecord` 读写 |
| `hybridsim_schedule_driver.py` | 离线驱动 HS `schedule_step` |
| `vllm_schedule_driver.py` | 离线驱动真实 vLLM `Scheduler`（CPU，假 output） |
| `compare.py` | 逐步 diff |
| `run_case.py` | CLI |
| `tests/test_schedule_alignment.py` | unittest：每个 case 两个 `subTest` |

### 3.3 Case 输入

`cases/<name>.json` 示例（`chunked_prefill`）：

```json
{
  "scheduler": {
    "max_num_scheduled_tokens": 64,
    "max_num_running_reqs": 8,
    "tokens_per_step": 8,
    "long_prefill_token_threshold": 8,
    "num_gpu_blocks": 256,
    "block_size": 16
  },
  "requests": [
    {
      "request_id": "1",
      "arrive_step": 0,
      "num_prefill_tokens": 24,
      "num_decode_tokens": 3,
      "prompt_base": 100
    }
  ]
}
```

要点：

- **到达用逻辑 step**（`arrive_step`），不是仿真秒。step 0 入队的请求在第 0 次 `schedule()` 前已经在 waiting 里。
- 未给 `prompt_token_ids` 时，两边用同一规则合成：`[prompt_base + i for i in range(prefill)]`。开 APC 的 case 必须显式给相同 token 列表。
- `framework` 默认 `"vllm"`，走 `SchedulerFactory`。

### 3.4 hybridsim driver 怎么跑

`run_hybridsim_schedule`（`asyncio`，因为 `schedule_step` 是 async）：

1. `SchedulerFactory.create("vllm", ...)` + `VllmKvCacheManager(num_gpu_blocks, block_size, enable_prefix_caching)`。
2. 循环 `step = 0 .. max_steps`：
   - 把 `arrive_step <= step` 的请求建成 `InferenceRequest`，`status=WAITING`，入 waiting。
   - `await schedule_step(...)`，`remote_lookup=None`。
   - 从 `ScheduleBatch.tokens_per_request` 抽出本步 ledger。
   - **立刻** `on_batch_complete`（零时长假执行）。
   - 记录 `scheduled_tokens` / `preempted_ids` / `finished_ids` / 队列 / KV 快照。
3. 无 waiting、无 running、无未来到达则停；连续空闲 3 步也停。

`PYTHONHASHSEED=0`，并 `reset_none_hash()`，让无 `hash_ids` 时的 block hash 与 vLLM `sha256` pickle 链一致。

### 3.5 vLLM driver 怎么跑

`run_vllm_schedule` **构造真实 `Scheduler`，但不启 Engine、不跑模型**：

1. `VLLM_TARGET_DEVICE=cpu`，`HF_HUB_OFFLINE=1`，模型用本地 `dummy_hf_model`（GPT2 配置，`skip_tokenizer_init=True`）。
2. 把 case 旋钮映射到 `SchedulerConfig` / `CacheConfig` / `KVCacheConfig`（`num_blocks` 与 HS `num_gpu_blocks` 同一值）。
3. 同一 step 循环：`arrive_step` 到点则 `scheduler.add_request`。
4. `output = scheduler.schedule()`。
5. 用 `_fake_output` 造 `ModelRunnerOutput`：若该请求本步后 `num_computed_tokens >= prompt_len`，采样 `[1]`，否则 `[]`。然后 `scheduler.update_from_output(output, fake)`。
6. 从 `output.num_scheduled_tokens`、`preempted_req_ids`、finish 前后差集抽出与 HS 同构的 ledger。
7. APC case：对新准入请求用 `num_computed_tokens - scheduled_n` 反推 `prefix_hit_tokens`。

依赖：torch + 本地 vLLM 树（`VLLM_ROOT`，默认 `/home/y_luchenda/vllm-main`）。

假 output 是对齐能成立的关键：vLLM `schedule()` 已把 computed 加上本步 scheduled 量；driver 只模拟「prompt 算完的那一步会多一个 sampled token」，与 HS `on_batch_complete` 的 +1 规则对齐。

### 3.6 Ledger 与比对

每步一条 `ScheduleStepRecord`：

| 字段 | 含义 | 默认是否比对 |
|------|------|----------------|
| `scheduled_tokens` | `req_id → 本步 token 数` | 是 |
| `preempted_ids` | 本步被 FCFS 抢占的请求 | 是 |
| `finished_ids` | 本步新完成 | 是 |
| `waiting_ids` / `running_ids` | 步后队列 | 否（`compare_queues`） |
| `free_blocks` / `allocated_blocks` / `prefix_hit_tokens` | KV / APC 快照 | 仅 `enable_prefix_caching=true` 的 case（`compare_kv`） |

`compare_ledgers`：

1. `filter_nonempty`：丢掉「无 scheduled、无 preempt/finish/new/prefix_hit」的空闲步，再把剩余步重编号。两边空闲拍数可以不同，只比有决策的步。
2. 逐步比上述字段；id 做字符串规范化。
3. 不等则 `CompareReport` 列出最多 20 条 diff。

因此对齐的是 **决策序列**，不是「第几个 DES tick」。

### 3.7 测试入口

单元测试 `tests/test_schedule_alignment.py` 对每个 case 跑两组：

1. **`TestScheduleAlignmentExpected`**：HS ledger vs 仓库里的 `*.expected.ledger.jsonl`（防 HS 自己回归）。
2. **`TestScheduleAlignmentVllm`**：HS ledger vs 当场跑的 vLLM ledger（防与上游语义漂）。

```bash
cd hybridsim
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONHASHSEED=0 \
  PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

单 case CLI：

```bash
PYTHONPATH=src/python:tests:. python -m schedule_alignment.run_case --case multi_decode
```

默认必须能跑 vLLM；`--skip-vllm` 只跑 HS。改语义后若 golden 需要更新：`--write-expected`。

### 3.8 当前 case 在验证哪条调度规则

| Case | 主要打到的流程 | 在看什么 |
|------|----------------|----------|
| `chunked_prefill` | D：chunk | prompt=24、阈值=8 → 三步 8+8+8 prefill，再 1+1 decode（最后 prefill 步采样 +1） |
| `multi_decode` | D + E：并发 decode | 三条短请求同时 decode，受 `max_num_scheduled_tokens=8` |
| `mixed_batch` | C：Phase1+2 同拍 | 已在跑的 decode 与后到的 prefill 抢同一 budget |
| `budget_exhaust` | C：Phase1 耗尽 budget | `max_num_scheduled_tokens=4`，Phase1 用完后同步不再 admit waiting |
| `preempt_oom` | E：null block + full-ISL + 自抢占 | `num_gpu_blocks=3`（可用 2 块）；第二条不能因「第一 chunk 很小」被准入 |
| `local_prefix` | E.2 关 APC 基线 | 同 prompt 也两侧全量 prefill |
| `local_prefix_hit` | E.2 开 APC | 共享 32 token、block=16；命中上限 `prompt-1` 再按块对齐 → hit 16，第二请求少一步 prefill |
| `local_prefix_partial` | E.2 非整块前缀 | 共享 24 token → 只计 1 个满块（16），剩余 8 仍要算 |

`chunked_prefill` 的 golden 逐步形态（过滤前）：

```text
step 0  scheduled {1: 8}   准入 + 第一 chunk prefill
step 1  scheduled {1: 8}
step 2  scheduled {1: 8}   prompt 24 结束
step 3  scheduled {1: 1}   完成 prefill 当步 +1 output
step 4  scheduled {1: 1}   最后 decode，finished {1}
```

### 3.9 已知差距（有意不声称对齐）

- Cluster 负载均衡、PD handoff、真实执行时长。
- Store / 远程 KV：见 `tests/mooncake_alignment/`。
- 公开 `hash_ids` trace 与 vLLM APC hasher（hasher 不同，alignment 只用同一套 `prompt_token_ids`）。
- Speculative decode、LoRA、encoder budget、`watermark>0`、vLLM async scheduling。
- HS `is_finished` 用 `computed >= prefill+decode`；vLLM 用 sampling / `max_tokens`。短 case 下两者一致，长尾停止条件未宣称逐比特相同。

---

## 4. 读代码顺序

1. `schedulers/factory.py` — `InferenceScheduler` 两个方法。
2. `schedulers/vllm_schedule.py` — 流程 C–F。
3. `actors/replica.py` 的 `on_request` / `on_step` / `on_batch_end` — DES 如何调用同一套 `schedule_step`。
4. `actors/cluster.py` + `cluster/monolith.py` / `pd.py` — 流程 A。
5. `tests/schedule_alignment/hybridsim_schedule_driver.py` 与 `vllm_schedule_driver.py` — 双端一步一账。
6. 对照：`vllm/v1/core/sched/scheduler.py`（`schedule` / `update_from_output`）。

扩展其他框架：实现 `InferenceScheduler`，`SchedulerFactory.register("sglang", ...)`，再写对应的 `*_schedule_driver` 抽同构 ledger。校准的是决策序列，不是把上游 Engine 嵌进仿真器。
