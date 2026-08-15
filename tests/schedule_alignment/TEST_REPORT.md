# Schedule Alignment 测试报告

日期：2026-08-11  
命令：

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONHASHSEED=0 \
  PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## 环境

| 项 | 值 |
|----|-----|
| `VLLM_ROOT` | `/home/y_luchenda/vllm-main` |
| Device | CPU（`VLLM_TARGET_DEVICE=cpu`，无 CUDA kernel） |
| HF | `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` |
| Hash | `PYTHONHASHSEED=0`（HS `block_keys` ↔ vLLM `sha256` pickle 链） |
| 模型 fixture | `tests/schedule_alignment/dummy_hf_model`（GPT2 配置） |

`tests/test_schedule_alignment.py`：**PASS**（expected golden + vLLM compare，含 KV 字段的 prefix case）。  
`tests/test_hash_ids_kv.py`：**PASS**。

---

## Part A — 调度决策（全部 case）

比对字段：`scheduled_tokens` / `preempted_ids` / `finished_ids`（空闲步已过滤）。

| Case | 描述 | nonempty steps | HS vs expected | HS vs vLLM |
|------|------|----------------|----------------|------------|
| `chunked_prefill` | 长 prompt 分 chunk | 5 | PASS | PASS |
| `multi_decode` | 多请求 decode | 5 | PASS | PASS |
| `mixed_batch` | prefill+decode 混合 | 5 | PASS | PASS |
| `budget_exhaust` | token budget 耗尽 | 7 | PASS | PASS |
| `preempt_oom` | null block + full-ISL + 自抢占 | 3 | PASS | PASS |
| `local_prefix` | 关缓存，同 prompt 全量 prefill | 4 | PASS | PASS |
| `local_prefix_hit` | 整块共享前缀 APC | 5 | PASS | PASS |
| `local_prefix_partial` | 非整块前缀只计 full blocks | 5 | PASS | PASS |

**结论**：调度输出（scheduled / preempt / finish）在 hybridsim `VllmScheduler.schedule_step` 与 offline vLLM `Scheduler.schedule` 上一致。

---

## Part B — KV / Prefix 逐步证据

`compare_kv=True` 仅对 `enable_prefix_caching=true` 的 case 开启；比对 `free_blocks`、`allocated_blocks`、`prefix_hit_tokens`。

### B.1 `local_prefix_hit`（prompt=32，block_size=16）

请求 1 完成后缓存 2 个 full block；请求 2 到达时 APC 命中受 vLLM 规则限制：`max_hit = prompt_len - 1` → block-align → **16**（不能 32，需重算最后 token）。

| step | sched HS/vLLM | prefix_hit | free_blocks | allocated_blocks |
|------|---------------|------------|-------------|------------------|
| 0 | `{1:16}` | — | 254 | `{1:1}` |
| 1 | `{1:16}` | — | 253 | `{1:2}` |
| 2 | `{1:1}` finish 1 | — | 255 | `{}` |
| 3 | `{2:16}` | `{2:16}` | 253 | `{2:2}` |
| 4 | `{2:1}` finish 2 | — | 255 | `{}` |

相对无缓存：请求 2 的 prefill 从两步 ×16 降为一步 ×16（命中 16 + 调度剩余 16）。

**KV compare：PASS**

### B.2 `local_prefix_partial`（prompt=24，block_size=16）

共享前缀非整块；可缓存 1 个 full block。请求 2：`prefix_hit=16`，剩余 prefill `8`。

| step | sched HS/vLLM | prefix_hit | free_blocks | allocated_blocks |
|------|---------------|------------|-------------|------------------|
| 0 | `{1:16}` | — | 254 | `{1:1}` |
| 1 | `{1:8}` | — | 253 | `{1:2}` |
| 2 | `{1:1}` finish 1 | — | 255 | `{}` |
| 3 | `{2:8}` | `{2:16}` | 253 | `{2:2}` |
| 4 | `{2:1}` finish 2 | — | 255 | `{}` |

**KV compare：PASS**

### B.3 关缓存基线 `local_prefix`

`enable_prefix_caching=false`：两侧均全量 prefill，无 `prefix_hit_tokens`；ledger 仍记录 `free_blocks` / `allocated_blocks` 供人工查看，默认不强制 KV 字段比对。

---

## 实现要点（本轮）

1. **本地 APC**：无 `hash_ids` 时用 `resolve_block_keys` / vLLM-compatible block hash；有 `hash_ids` 仍走 trace keys。  
2. **命中上限**：与 vLLM `get_computed_blocks` 一致（`prompt_len - 1` + block 对齐）。  
3. **Ledger**：`ScheduleStepRecord` 增加 `free_blocks` / `allocated_blocks` / `prefix_hit_tokens`；`ScheduleResult.prefix_hits` 由 `process_wait_queue` 返回。  
4. **文档**：`schedule_alignment.md` / `README.md` 已更新 post-`hash_ids` 与 APC 说明。

## 有意不纳入

- Cluster 分发、真实 GPU、Mooncake 远程时序  
- Spec decode / LoRA / encoder budget  
- 公开 remapped `hash_ids` trace 与 vLLM APC hasher 直接混比
