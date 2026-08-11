# KV Cache Trace Sources (kvcache-simulator / KV Cache Lab)

调研依据：[`kvcache-ai/kvcache-blog`](https://github.com/kvcache-ai/kvcache-blog) 的
`data/kv_cache_lab/presets.yaml` 与 `scripts/lib/kv-cache-lab-traces.mjs`
（网页 Hit Rate Simulator 的官方 preset 列表）。

## 关键结论

1. **网页工具的“内置 trace”不在仓库里提交 raw 文件**，而是按 `TRACE_SOURCES` 从公网下载到本地 cache，再归一化为 Mooncake JSONL schema。
2. **绝大多数公开 hash-trace 不含真实 `token_ids` / `input_ids`**（隐私脱敏后只保留 remapped `hash_ids` + 长度）。
3. 若 hybridsim 需要“真实 token + hash chain”：
   - **hash-only 公开集**：只能做 **prefix 共享结构等价**（由 `hash_ids` 合成 token，或 Store 直接吃 `hash_ids`）。
   - **有文本的 agent 集**（LMCache / Exgentic）：可 tokenize 得到真实 token，再算 hash chain。
   - **自有 SGLang 日志**：官方脚本 `scripts/sglang-log-to-kvcache-trace.py` 从 `input_ids=[...]` 生成带 `hash_ids` 的 JSONL（**同时保留真实 token 最可行**）。

## 统一目标 schema（kvcache-simulator / 网页上传）

每行一条请求（JSONL）：

```json
{"timestamp": 0.0, "block_size": 64, "hash_ids": [1, 2, 3], "input_length": 192, "output_length": 50}
```

| 字段 | 必需 | 含义 |
|------|------|------|
| `hash_ids` | 是 | 按前缀顺序的 block 身份（整数；同 id ⇒ 可共享该 block 的 KV） |
| `input_length` | 是 | prefill token 数（hit rate 分母） |
| `block_size` | 条件 | 源原生粒度；全 trace 一致 |
| `timestamp` / `output_length` | 否 | 到达时间 / decode 长度；sim 分母可忽略 output |

## Hit-rate 可用 preset（`sourceKind: hash` 或可导出 hash）

| id | 标签 | block | 下载入口 | 是否含真实 token | 备注 |
|----|------|-------|----------|------------------|------|
| `mooncake_fast25` | Mooncake FAST25 Tool&Agent | 512 | [arxiv-trace/mooncake_trace.jsonl](https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl) | **否** | 网页主用；全局 remapped hash |
| Mooncake FAST25 scenarios | conversation / synthetic / toolagent | 512 | [FAST25-release/traces](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces) | **否** | 论文场景拆分 |
| `bailian_qwen_trace_a` | Qwen Bailian Trace A | 16 | [qwen_traceA_blksz_16.jsonl](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceA_blksz_16.jsonl) | **否** | 含 `chat_id`/`parent_chat_id`/`type`/`turn` |
| `ragpulse` | RAGPulse | 512 | [flashserve/RAGPulse](https://huggingface.co/datasets/flashserve/RAGPulse) | **否** | `hash_ids` 按类别分桶（sys/passages/history/…） |
| `semianalysis_weka_no_subagents` | Weka Claude Code | 64 | [cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226) | **否** | session 内 `hash_id_scope=local` |
| `semianalysis_weka_with_subagents_256k` | Weka + Subagents | 64 | [cc-traces-weka-with-subagents-052726-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726-256k) | **否** | 含 sub-agent；需 namespace |
| `kv_cache_tester_claude_code` | kv-cache-tester | 64 | [callanjfox/kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) `traces/` | **否** | WEKA 风格 session JSON |

### 相关但非网页直接 URL 的镜像

- [valeriol29/mooncake-traces](https://huggingface.co/datasets/valeriol29/mooncake-traces) — Mooncake 官方 trace 的 HF 镜像  
- [wchen22/isb1-cc-traces](https://huggingface.co/datasets/wchen22/isb1-cc-traces) / [semianalysisai/cc-traces-weka-042026](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-042026) — 同类 Claude Code hash session  

## 文本型（无原生 hash_ids；工具离线估 64-token bucket）

| id | 数据集 | 内容 | 对 hybridsim 的价值 |
|----|--------|------|---------------------|
| `lmcache_agentic_sample` | [zeelHz/lmcache-agentic-traces](https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces) | `input` 消息文本 + `output_length` | **可 tokenize → 真实 token + 自建 hash chain** |
| `exgentic_agent_sample` | [Exgentic/agent-llm-traces](https://huggingface.co/datasets/Exgentic/agent-llm-traces) | agent span（payload 文本） | 同上，需从 span 抽文本 |

## 仅作到达/长度参考（不可直接做 hit-rate）

- [BurstGPT](https://huggingface.co/datasets/lzzmm/BurstGPT)  
- [SwissAI Serving Trace](https://huggingface.co/datasets/eth-easl/swissai-serving-trace)  

## 自有生产路径（推荐若必须真实 token）

官方转换器：`kvcache-blog/scripts/sglang-log-to-kvcache-trace.py`

- 输入：SGLang worker log 的 `Finish:` 行（含 `input_ids=[...]`）  
- 输出：Mooncake JSONL（`hash_ids` = blake2b 前缀累计 hash）  
- **可同时导出 `input_ids` 扩展字段**，满足 hybridsim Store（vLLM 链式 hash）与调度。

## 样本校验结果（本机 `samples/`）

| 样本 | 观察到的字段 | 真实 token？ |
|------|--------------|--------------|
| `mooncake_fast25.head.jsonl` | `timestamp,input_length,output_length,hash_ids` | 无 |
| `bailian.head.jsonl` | + `chat_id,parent_chat_id,type,turn` | 无 |
| `toolagent.head.jsonl` | 同 Mooncake | 无 |
| `weka_no_subagents.row0.json` | session + `requests[].hash_ids/in/out/t` | 无；hash **per-trace local** |
| `ragpulse.row0.json` | 分类 `hash_ids` dict | 无 |
| `lmcache.row0.json` | `input[].content` 文本 | **有文本，无 hash_ids** |
| `exgentic.row0.json` | `spans` | 文本在 attributes，无 hash_ids |

## 本地目录约定

```
data/kvcache_traces/
  README.md           # 使用说明 + generator 入口
  SOURCES.md          # 本文件（来源调研）
  blog/               # 从 kvcache-blog 拉下的 presets / scripts
  samples/            # schema 小样本
  raw/                # 完整 raw 下载（gitignore）
  normalized/         # 归一化 JSONL（gitignore；generator 默认读这里）
  meta/               # catalog / 下载状态
```

## 已下载到 `raw/`（本机）

| 文件 | 备注 |
|------|------|
| `mooncake_trace.jsonl` | FAST25 主 preset（normalized 为 `mooncake_fast25.jsonl`） |
| `mooncake_toolagent_trace.jsonl` / `conversation` / `synthetic` | FAST25 场景拆分 |
| `qwen_traceA/B_blksz_16.jsonl` 等 | Bailian |
| `kv_cache_tester/` | Weka Claude Code sessions |

详见 [`README.md`](README.md) 的 normalized 可用列表。

## hybridsim 对接现状

- `KvCacheTraceRequestGenerator` 读 `normalized/*.jsonl`，填入 `InferenceRequest.hash_ids` / `block_size` / 长度；缺 `input_ids` 时合成 placeholder tokens。
- **Store / 本地 APC**：`hash_ids` 非空时直接作 block key；否则 `block_keys_from_tokens(prompt_token_ids)`。
- 真实 token + hash：优先 SGLang log → 扩展 JSONL（保留 `input_ids`）；或 LMCache 文本 tokenize。
