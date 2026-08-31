# hybridsim 请求生成

请求进入仿真的统一实体是 `InferenceRequest`（`hybridsim_infer/request.py`）。  
三种生成方式都产出 `list[InferenceRequest]`，再经 `schedule_arrivals` / `schedule_from_generator` 注入 `ClusterActor`。

请求到达之后如何变成 batch（集群分发、replica `schedule_step`、与 vLLM 对照、对齐测试）：[`scheduler.md`](scheduler.md)。

---

## 1. `InferenceRequest` 数据结构

| 字段 | 类型 | 谁填 | 含义与仿真关系 |
|------|------|------|----------------|
| `request_id` | `int` | 生成器 / 手写 | 请求唯一标识。贯穿 Cluster 派发、Replica 队列、Engine 事件、request profile 元数据。 |
| `arrived_at` | `float` | 生成器 / 手写 | DES 到达时刻（秒）。`ClusterScheduler.schedule_arrivals` 用它 `send_at`，决定请求何时进入集群调度。 |
| `num_prefill_tokens` | `int` | 生成器 / 手写 | Prompt 长度。决定 prefill 阶段要算多少 token；与 `num_computed_tokens` 比较可判断是否仍在 prefill（`is_prefill_chunk`）。InferWorkloadGenerator / op-level 计时也依赖本步实际 prefill chunk 长度。 |
| `num_decode_tokens` | `int` | 生成器 / 手写 | 目标输出长度。与 prefill 一起构成请求总工作量；`is_finished()` 要求 `num_computed_tokens >= prefill + decode`。 |
| `num_computed_tokens` | `int` | **运行时**（生成时通常 0） | 已计算 token 数。本地 APC / Store 命中后会抬高，从而缩小本步要算的 chunk；调度、KV 拉取、analytical attention（`cached + chunk`）都读它。 |
| `num_output_tokens` | `int` | **运行时**（生成时通常 0） | 已采样输出长度。抢占后仍保留，用于 admit / `num_tokens`（对齐 vLLM「当前序列长度 = prompt + 已输出」）。 |
| `prompt_token_ids` | `list[int]` | 生成器 / `__post_init__` | 见下方专题。 |
| `hash_ids` | `list[int]` | 主要来自 KV trace | 见下方专题。 |
| `block_size` | `int` | KV trace / 参数 | 每个 `hash_id`（或链式 hash 一块）对应多少 token。Store / APC 按块对齐；Mooncake 常 512，Bailian 常 16，Weka 常 64。 |
| `status` | `RequestStatus` | **运行时** | `WAITING` / `RUNNING` / `WAIT_FOR_REMOTE_KVS` / `PREEMPTED` / `FINISHED`。Replica、KV pull、抢占逻辑围绕它流转。 |
| `completed` | `bool` | **运行时** | 请求是否已走完仿真生命周期（供 profile / 收尾）。 |
| `pending_remote_tokens` | `int` | **运行时** | 正在远程拉取的 KV token 数；与 `WAIT_FOR_REMOTE_KVS` 配合。 |
| `kv_transfer_params` | `dict \| None` | **Cluster（PD）** | Mooncake 风格控制面参数，如 `do_remote_decode` / `do_remote_prefill` + `remote_replica_id`。生成器不填；PD 拓扑在 arrive / handoff 时盖章。 |
| `pending_lookup` / `lookup_result` | 运行时 | Store 异步 lookup 进行中及缓存的回复；调度下一步再消费 hit 结果。 |

常用派生属性：

- `num_tokens` = prefill + 已输出（当前序列长）
- `num_tokens_with_output` = prefill + 目标 decode（完成所需总长）
- `remaining_tokens` / `is_finished()` / `is_prefill_chunk`

### `prompt_token_ids` 与 `hash_ids`（重点）

二者都可以描述「前缀长什么样」，但职责不同：

| | `prompt_token_ids` | `hash_ids` |
|--|-------------------|------------|
| 是什么 | tokenizer 意义上的 token 序列（或占位序列） | 按前缀顺序排列的 **KV block 身份** |
| 粒度 | 每个元素 ≈ 1 token | 每个元素 ≈ `block_size` 个 token 的一块 KV |
| 谁用 | Framework / 本地 APC 在需要「按 token 切片」时；无 `hash_ids` 时用 vLLM 同款链式 hash 算 Store key | Store / 本地 APC 在有权威块 id 时 **直接** 当 block key（`block_keys_from_hash_ids`） |
| 典型来源 | 手写、ServeGen（常由 `__post_init__` 按 `request_id` 合成）、SGLang 日志里的真实 `input_ids`、公开 trace 上的占位合成 | Mooncake / Bailian / Weka 等公开轨迹 |

关键规则：

1. **公开 KV 轨迹几乎没有真实 prompt**，只有脱敏后的 `hash_ids`。这时 `hash_ids` 是前缀共享的权威来源；`prompt_token_ids` 若存在，多半是长度占位，**不能**再拿去链式 hash 冒充 Store key。
2. **无 `hash_ids`、只有 token**（List / ServeGen / schedule_alignment）：Store / APC 走 `block_keys_from_tokens(prompt_token_ids)`，与 vLLM APC 对齐时需同一套真实 token。
3. 构造时若已有 `hash_ids`，`__post_init__` **不会**再按 `request_id` 造「每请求唯一」假 prompt，以免毁掉跨请求共享前缀。

一句话：仿真 **命中率 / Store 前缀** 看 `hash_ids`（有则优先）；**算力长度与调度进度** 看 `num_*_tokens` / `num_computed_tokens`；`prompt_token_ids` 在无权威 hash 时承担「可哈希的内容」，有权威 hash 时只作可选填充。

---

## 2. 基本用法

统一接线：

```python
from hybridsim_infer import InferenceConfig, build_inference_simulation

sim = build_inference_simulation(InferenceConfig(...))
sim.schedule_from_generator(gen)   # 或 sim.schedule_arrivals(reqs)
sim.run()
```

### 2.1 List：手写 / 单测 / Demo

**场景**：拓扑冒烟、固定到达时刻、精确控制 prompt 是否共享、不依赖外部数据。

```python
from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators import ListRequestGenerator

reqs = [
    InferenceRequest(request_id=0, arrived_at=0.0, num_prefill_tokens=128, num_decode_tokens=32),
    InferenceRequest(request_id=1, arrived_at=0.5, num_prefill_tokens=64, num_decode_tokens=16),
]
sim.schedule_from_generator(ListRequestGenerator(reqs))
```

也可直接 `sim.schedule_arrivals(reqs)`。无 `hash_ids` 时会按 `request_id` 合成互不相同的假 `prompt_token_ids`（适合测「不共享」）；要测共享前缀则自己设相同的 `prompt_token_ids` 或相同的 `hash_ids`。

### 2.2 ServeGen：合成到达过程 + 长度分布

**场景**：需要「像线上一样」的到达强度与 input/output 长度分布，但不关心真实前缀共享结构（无 `hash_ids`）。

官方库：[alibaba/ServeGen](https://github.com/alibaba/ServeGen)。基本原理：

1. `ClientPool` 加载某类负载的统计模型（hybridsim 当前仅 `category='language'`，读 ServeGen 旁的 `data/language/<model>`）。
2. 给定时间窗与到达率函数 `rate_fn`（可常数或自定义曲线），`generate_workload` 采样出带 `timestamp`、`input_tokens`、`output_tokens` 的请求序列。
3. hybridsim 的 `ServeGenRequestGenerator` 把上述字段映射为 `arrived_at` / `num_prefill_tokens` / `num_decode_tokens`。

```python
from hybridsim_infer.request_generators import ServeGenRequestGenerator

gen = ServeGenRequestGenerator(
    category="language",
    model="m-small",
    duration=60,
    rate=5.0,
    seed=0,
    max_requests=100,
)
sim.schedule_from_generator(gen)
```

可选依赖：`pip install -e <ServeGen克隆>` 或 `pip install -e ".[servegen]"`。示例：`examples/inference/servegen_demo.py`。

ServeGen 在本项目中 **只做请求到达/长度**，不做 GPU 时长；时长仍由 `InferWorkloadGenerator`（batch_level / op_level）决定。

### 2.3 KV Cache Trace：真实前缀共享结构

**场景**：仿真 Store / 本地 APC **命中率**、前缀共享对调度与 KV 流量的影响。数据带权威 `hash_ids`。

数据目录：`src/python/hybridsim_infer/request_generators/kvcache_traces/`（常量 `KVCACHE_TRACES_DIR`）。

```text
raw/           # 公网原始文件（gitignore）
normalized/    # 统一 JSONL，generator 读这里（gitignore）
samples/       # 小样本
meta/          # catalog.json
```

```python
from hybridsim_infer.request_generators import (
    KVCACHE_TRACES_DIR,
    KvCacheTraceRequestGenerator,
)

gen = KvCacheTraceRequestGenerator(
    KVCACHE_TRACES_DIR / "normalized" / "mooncake_fast25.jsonl",
    block_size=512,
    max_requests=1000,
)
sim.schedule_from_generator(gen)
```

刷新归一化：`python3 tools/normalize_kvcache_traces.py`。

#### 数据来源与处理流程

统一目标：一行一条请求的 Mooncake 风格 JSONL，再由 `map_kvcache_trace_record` 变成 `InferenceRequest`。

```text
原始数据 →（按来源归一化）→ normalized/*.jsonl → KvCacheTraceRequestGenerator → InferenceRequest
```

**1）Mooncake FAST25**  
来源：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) `FAST25-release`（如 `arxiv-trace/mooncake_trace.jsonl`）。  
原始已是 JSONL：`timestamp, input_length, output_length, hash_ids`（block 常 512）。  
处理：拷入 `raw/` → normalize 补全 `block_size=512` → 写入 `normalized/mooncake_fast25.jsonl` → generator 映射为 prefill/decode/`hash_ids`/`arrived_at`；无真实 token 时按 hash 合成占位 `prompt_token_ids`。

**2）Qwen Bailian**  
来源：[alibaba-edu/qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)。  
原始同样是带 `hash_ids` 的 JSONL（block 常 16），可带 `chat_id` 等会话字段。  
处理：与 Mooncake 相同链路，normalize 写入 `block_size=16`（如 `qwen_bailian_traceA.jsonl`）。前缀命中主要用 `hash_ids` + 长度，会话元数据不参与 key。

**3）Weka / kv-cache-tester**  
来源：[callanjfox/kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)（session JSON；与 SemiAnalysis Weka 同类）。  
原始是 **按 session 的 JSON**：`requests[].{hash_ids, in, out, t}`，且 hash **在 session 内局部编号**（不同 session 的 `0` 不是同一块）。  
处理：`flatten_weka_sessions` 展平每一请求为一行，把 `(session_id, local_hash)` **remap 成全局整数**，`in/out/t` → `input_length/output_length/timestamp`，写出 `normalized/weka_claude_code_kv_cache_tester.jsonl`（block 常 64）→ 再进 generator。

**4）自有 SGLang 日志（可选，要真实 token 时）**  
来源：SGLang worker `Finish:` 日志中的 `input_ids=[...]`。  
处理：用 [kvcache-blog 的转换脚本](https://github.com/kvcache-ai/kvcache-blog/blob/main/scripts/sglang-log-to-kvcache-trace.py) 生成带 `hash_ids` 的 JSONL；若 JSONL 保留 `input_ids`，generator **优先用真实 token**，Store 也可改走 vLLM 链式 hash（与公开 remapped hash 路径不同，勿混比）。

Generator 字段映射（所有来源归一化之后相同）：

| JSONL | `InferenceRequest` |
|-------|-------------------|
| `timestamp` / `t` | `arrived_at`（可再 `time_scale` / `time_offset`） |
| `input_length` / `in` | `num_prefill_tokens` |
| `output_length` / `out` | `num_decode_tokens` |
| `hash_ids` | `hash_ids`（权威 block key） |
| `block_size` | `block_size` |
| `input_ids`（若有） | `prompt_token_ids`；否则可合成占位 |

默认 `require_hash_ids=True`：无 hash 的行跳过。生成后按到达时间排序并重编号 `request_id`。
