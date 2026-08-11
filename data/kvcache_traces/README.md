# KV Cache 轨迹数据（hybridsim）

本目录存放供 [`KvCacheTraceRequestGenerator`](../../src/python/hybridsim_infer/request_generators/kvcache_trace_generator.py) 使用的公开 KV 前缀轨迹，格式对齐 [KVCache.AI Hit Rate Simulator](https://kvcache.ai/tools/kv-cache-hit-rate-simulator/) / [`kvcache-simulator`](https://github.com/kvcache-ai/kvcache-blog/tree/main/packages/kvcache-simulator)。

用途：按真实业务风格的 **前缀共享结构** 驱动 hybridsim 请求流，仿真 Store / 本地前缀命中率。  
来源调研与 preset 清单见 [`SOURCES.md`](SOURCES.md)。

---

## 关键限制

公开 preset **几乎都没有真实 `input_ids` / token**，只有隐私脱敏后的：

- `hash_ids`：按前缀顺序排列的 block 身份（整数；同 id ⇒ 可共享该块 KV）
- `input_length` / `output_length`：prefill / decode 长度
- `block_size`：每个 hash 对应多少 token

因此：

- **Store / 命中仿真**：以 `hash_ids` 为权威 block key（不要对占位 token 再做 vLLM 链式 hash）。
- **调度仍可能需要 token 列表**：generator 会按长度合成 **占位** `prompt_token_ids`，仅作长度填充，**不代表原文**。

若需要「真实 token + hash」：用 SGLang `Finish:` 日志经 [sglang-log-to-kvcache-trace.py](https://github.com/kvcache-ai/kvcache-blog/blob/main/scripts/sglang-log-to-kvcache-trace.py) 转换，并在 JSONL 中保留 `input_ids`（generator 已支持）。

---

## 目录结构

| 路径 | 内容 |
|------|------|
| `raw/` | 下载的原始文件（Mooncake、Bailian、kv-cache-tester 等；gitignore） |
| `normalized/` | 归一化后的扁平 JSONL，**generator 默认读这里**（gitignore） |
| `meta/catalog.json` | 来源与归一化统计 |
| `samples/` | 小样本，便于肉眼看 schema |
| `blog/` | kvcache-blog 上游 presets / 脚本快照 |
| `SOURCES.md` | 数据源调研 |

刷新归一化文件：

```bash
python3 tools/normalize_kvcache_traces.py
```

---

## 公开数据 → hybridsim 请求：完整链路

```text
公网原始 trace
    │
    ▼
raw/                    # 原样落盘
    │
    │  tools/normalize_kvcache_traces.py
    ▼
normalized/*.jsonl      # 统一 Mooncake 风格行记录
    │
    │  KvCacheTraceRequestGenerator / map_kvcache_trace_record
    ▼
list[InferenceRequest]  # hybridsim 仿真请求
```

### 1. 公开数据长什么样

**Mooncake / Bailian（已是 JSONL）** 一行一例：

```json
{
  "timestamp": 0.0,
  "input_length": 192,
  "output_length": 50,
  "hash_ids": [1, 2, 3],
  "block_size": 512
}
```

含义：

| 字段 | 含义 |
|------|------|
| `hash_ids` | 前缀块身份链；长度 ≈ `ceil(input_length / block_size)`（末块可为 partial） |
| `input_length` | prompt token 数（hit-rate 分母） |
| `output_length` | decode 长度 |
| `block_size` | 每块 token 数（Mooncake 常为 512，Bailian 常为 16） |
| `timestamp` | 到达时间（秒或相对时间） |

Bailian 还可能带 `chat_id` / `parent_chat_id` / `type` / `turn` 等会话元数据；**归一化与 generator 做前缀命中时主要用 `hash_ids` + 长度**。

**Weka / kv-cache-tester（按 session 的 JSON）** 大致为：

```json
{
  "id": "session-xxx",
  "block_size": 64,
  "requests": [
    {"hash_ids": [0, 1, 2], "in": 128, "out": 32, "t": 1.5}
  ]
}
```

注意：session 内 `hash_ids` 往往是 **局部命名空间**（不同 session 的 `0` 不是同一块）。归一化时会 remap 成全局 id。

### 2. 归一化（`normalize_kvcache_traces.py`）

对 Mooncake / Bailian：

- 读 `raw/<源文件>`，写出 `normalized/<名字>.jsonl`
- 若行内缺 `block_size`，写入该源约定值（512 或 16）
- 其它字段原样保留

对 Weka session 目录：

- 遍历 `raw/kv_cache_tester/*.json`
- 每个 session 的每个 request 展平为一行 JSONL
- `(session_id, local_hash)` → 全局递增整数，保证跨 session 可比
- 字段映射：`in`→`input_length`，`out`→`output_length`，`t`→叠加后的 `timestamp`

归一化后的统一行格式（generator 输入）：

```json
{
  "timestamp": 0.0,
  "block_size": 64,
  "hash_ids": [2001, 2002],
  "input_length": 128,
  "output_length": 32
}
```

可选扩展：`"input_ids": [...]` / `"prompt_token_ids": [...]`（有真实 token 时）。

### 3. 映射为 `InferenceRequest`（`map_kvcache_trace_record`）

| 公开 JSONL 字段 | `InferenceRequest` 字段 | 说明 |
|-----------------|-------------------------|------|
| `input_length`（或 `in`） | `num_prefill_tokens` | 必填；≤0 则跳过该行 |
| `output_length`（或 `out`） | `num_decode_tokens` | 缺省为 0 |
| `timestamp`（或 `t`） | `arrived_at` | 可再乘 `time_scale`、加 `time_offset` |
| `hash_ids` | `hash_ids` | Store / 本地 APC 的权威 block key 链 |
| `block_size` | `block_size` | 与 `hash_ids` 粒度一致 |
| `input_ids` / `prompt_token_ids` | `prompt_token_ids` | **有则优先用真实 token** |
| （无真实 token 时） | `prompt_token_ids` | 由 `hash_ids` **合成占位序列**（见下） |
| （生成顺序） | `request_id` | 排序后按到达时间重编号 |

占位 `prompt_token_ids` 规则（`synthesize_prompt_tokens=True`，默认开）：

1. 按每个 `hash_id` 填最多 `block_size` 个整数，总长凑满 `input_length`
2. 数值由 `hash_id` 派生，保证「不同块 id → 不同占位片段」，方便仍依赖 `prompt_token_ids[:n]` 的代码路径
3. **不可还原原文**；有 `hash_ids` 时 Store **不得**再对这串占位 token 做 vLLM sha256 链

`InferenceRequest.__post_init__`：若已有 `hash_ids`，**不会**再按 `request_id` 造一套唯一假 prompt（避免毁掉共享前缀）。

### 4. 进仿真后 key 怎么用

```text
resolve_block_keys / KvClient / 本地 APC：
  有 hash_ids  → 直接 stringify 成 Store/APC key（权威）
  无 hash_ids  → block_keys_from_tokens(prompt_token_ids)（vLLM 同款链式 hash）
```

因此本目录公开数据的正确用法是：**带着 `hash_ids` 跑 Store / hit-rate**。  
若要与真实 vLLM APC 字节级对齐，应改用同一套真实 `prompt_token_ids`（见 `tests/schedule_alignment`），不要混用公开 remapped `hash_ids`。

---

## 当前可用（normalized）

| 文件 | block_size | 约请求数 | 来源 |
|------|-----------|----------|------|
| `normalized/mooncake_fast25.jsonl` | 512 | 2.4 万 | [Mooncake FAST25](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/arxiv-trace) |
| `normalized/qwen_bailian_traceA.jsonl` | 16 | 4.3 万 | [Qwen Bailian](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) |
| `normalized/qwen_bailian_traceB.jsonl` | 16 | 17 万 | 同上 |
| `normalized/qwen_bailian_coder.jsonl` | 16 | 4.3 万 | 同上 |
| `normalized/qwen_bailian_thinking.jsonl` | 16 | 1.1 万 | 同上 |
| `normalized/weka_claude_code_kv_cache_tester.jsonl` | 64 | ~5.9 万（739 session） | [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) |

`raw/` 另有 Mooncake conversation / synthetic / toolagent 场景拆分；若要纳入 normalized，可在 `tools/normalize_kvcache_traces.py` 中加映射条目后重跑。

## 因 HF 代理未下到的集

- [RAGPulse](https://huggingface.co/datasets/flashserve/RAGPulse)
- [SemiAnalysis Weka](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226)（可用 kv-cache-tester 替代）
- [LMCache agentic](https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces)（文本型，可 tokenize 后再建 hash）
- [Exgentic agent](https://huggingface.co/datasets/Exgentic/agent-llm-traces)

---

## 使用示例

```python
from hybridsim_infer.request_generators import KvCacheTraceRequestGenerator

gen = KvCacheTraceRequestGenerator(
    "data/kvcache_traces/normalized/mooncake_fast25.jsonl",
    max_requests=1000,
)
reqs = gen.generate()

# 每条请求：
#   req.hash_ids          # 权威 block 链
#   req.block_size        # 如 512
#   req.num_prefill_tokens / req.num_decode_tokens
#   req.prompt_token_ids  # 占位或真实 input_ids
#   req.arrived_at
```

常用构造参数：

| 参数 | 含义 |
|------|------|
| `max_requests` | 最多读多少条 |
| `block_size` | 行内缺省时的默认块大小 |
| `time_scale` / `time_offset` | 缩放/平移到达时间 |
| `synthesize_prompt_tokens` | 无 `input_ids` 时是否合成占位 token（默认 True） |
| `require_hash_ids` | 无 `hash_ids` 的行是否跳过（默认 True） |
