# Schedule alignment (hybridsim ↔ vLLM)

测试套件：同输入下对齐 **replica 内 schedule 输出**（不关注 cluster 分发、不跑真实 GPU）。
设计意图见 `hybridsimdesign/hybridsim inference offline校准.md`。

**调度实现（请求 → batch，含本套件怎么跑）**：[`docs/scheduler.md`](../../docs/scheduler.md)  
**代码级对齐说明**：[schedule_alignment.md](schedule_alignment.md)  
**回归报告**：[TEST_REPORT.md](TEST_REPORT.md)

| File | Role |
|------|------|
| `schedule_alignment.md` | hybridsim ↔ vLLM 调度语义与代码映射 |
| `schema.py` / `compare.py` | Step ledger + diff（含可选 KV/prefix 字段） |
| `hybridsim_schedule_driver.py` | Drive `SchedulerFactory` → `schedule_step` offline |
| `vllm_schedule_driver.py` | Real vLLM `Scheduler` offline (fake `ModelRunnerOutput`, CPU, no GPU) |
| `cases/*.json` | Shared fixtures |
| `cases/*.expected.ledger.jsonl` | Golden hybridsim ledgers |
| `run_case.py` | CLI（vLLM compare **required** unless `--skip-vllm`） |
| `../test_schedule_alignment.py` | 单元测试入口（每个 case 为 `subTest`） |

## Run tests

```bash
cd hybridsim
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## Run a single case (CLI)

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONHASHSEED=0 PYTHONPATH=src/python:tests:. \
  python -m schedule_alignment.run_case --case multi_decode
```

Requires torch + local vLLM tree (`VLLM_ROOT`, default `/home/y_luchenda/vllm-main`).

## Aligned behaviors

- Chunked prefill / token budget / multi-request decode
- Null KV block reserved (vLLM-like)
- `scheduler_reserve_full_isl` admission gate
- FCFS preemption including self-preempt on OOM
- Decode default 1 token/step; last prefill forward samples +1 output token
- Local APC via vLLM-compatible block hashes when `enable_prefix_caching=true`
  (`prefix_hit_tokens` / `free_blocks` / `allocated_blocks` on prefix cases)

## hash_ids / APC 现状

- 仿真侧请求可带 `hash_ids`（Store / Mooncake trace）；alignment harness 以 `prompt_token_ids` 为主。
- 无 `hash_ids` 时 HS 本地 APC 使用 `resolve_block_keys`（与 vLLM `sha256` pickle 链一致，需 `PYTHONHASHSEED=0`）。
- `local_prefix` 仍关缓存作全量 prefill 基线；`local_prefix_hit` / `local_prefix_partial` 开启 APC。

## Known gaps

- No speculative decode / LoRA / encoder budget
- No cluster / GPU / remote Mooncake timing in this suite
- Public remapped `hash_ids` traces are not compared directly to vLLM APC hasher
