# Schedule alignment (hybridsim ↔ vLLM)

测试套件：同输入下对齐 **replica 内 schedule 输出**（不关注 cluster 分发、不跑真实 GPU）。
设计意图见 `hybridsimdesign/hybridsim inference offline校准.md`。

**代码级对齐说明（推荐先读）**：[schedule_alignment.md](schedule_alignment.md)

| File | Role |
|------|------|
| `schedule_alignment.md` | hybridsim ↔ vLLM 调度语义与代码映射 |
| `schema.py` / `compare.py` | Step ledger + diff |
| `hybridsim_schedule_driver.py` | Drive `FrameworkFactory` → `schedule_step` offline |
| `vllm_schedule_driver.py` | Real vLLM `Scheduler` offline (fake `ModelRunnerOutput`, CPU, no GPU) |
| `cases/*.json` | Shared fixtures |
| `cases/*.expected.ledger.jsonl` | Golden hybridsim ledgers |
| `run_case.py` | CLI（vLLM compare **required** unless `--skip-vllm`） |
| `../test_schedule_alignment.py` | 单元测试入口（每个 case 为 `subTest`） |

## Run tests

```bash
cd hybridsim
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python tests/test_schedule_alignment.py -v
```

## Run a single case (CLI)

```bash
HF_HUB_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=src/python:tests:. \
  python -m schedule_alignment.run_case --case multi_decode
```

Requires torch + local vLLM tree (`VLLM_ROOT`, default `/home/y_luchenda/vllm-main`).

## Aligned behaviors

- Chunked prefill / token budget / multi-request decode
- Null KV block reserved (vLLM-like)
- `scheduler_reserve_full_isl` admission gate
- FCFS preemption including self-preempt on OOM
- Decode default 1 token/step; last prefill forward samples +1 output token

## Known gaps

- HS prefix cache is token-list match, not vLLM APC hashes (off by default via `enable_prefix_caching`)
- No speculative decode / LoRA / encoder budget
