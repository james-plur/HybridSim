# Workload Generator ↔ Frontier RF 对齐报告

## 1. 设计概览

hybridsim 的 workload generator 把调度结果 `ScheduleBatch` 变成 Engine 可执行的 TimeoutKernel workload，有两条路径：

| 路径 | `duration_mode` | Generator | 产出 |
|------|-----------------|-----------|------|
| batch_level | `batch_level` + `batch_predictor`（`fixed` / `token_proportional` / `frontier`） | `BatchLevelWorkloadGenerator` | **1** 个 TimeoutKernel（整 batch 时长） |
| op_level | `op_level` | `OpLevelWorkloadGenerator` | **多层 Operator DAG** → 多个 TimeoutKernel（Roofline / α-β） |

工厂入口：`workload_generators/factory.py`（`make_infer_workload_generator`）。

- **frontier**：委托 Frontier `BaseExecutionTimePredictor`（通常 RandomForest）估 `ExecutionTime.total_time`。
- **op_level**：mock `Module.forward` 展开与 Frontier 算子名对齐的 DAG，再对每个 op 做解析计时；依赖正确时自然支持计算与通信交叠。后续也可把 op 交给底层 kernel（当前未做）。

---

## 2. Frontier 怎么估时

### 2.1 hybridsim 适配层

`FrontierBatchDurationPredictor`（`predictors/frontier.py`）把 `ScheduleBatch` 适配为 Frontier `Batch`：

| hybridsim | Frontier |
|-----------|----------|
| `req.num_prefill_tokens` / `num_decode_tokens` | 同名字段 |
| `req.num_computed_tokens` | `Request.num_processed_tokens` |
| `tokens_per_request` / chunk `num_tokens` | `Batch.num_tokens`（本步要算的 token） |
| `computed >= prefill` | 强制 `_is_prefill_complete = True`（decode） |

然后调用 `predict_stage_execution_time(...).total_time`。

### 2.2 Dense attention 特征（非 dummy RF）

| 算子 | 特征 |
|------|------|
| `attn_kv_cache_save` | `num_tokens`（本步写回长度） |
| `attn_prefill` | `kv_cache_size`, `prefill_chunk_size_squared` |
| `attn_decode` | `batch_size`, `kv_cache_size` |

运行时抽取逻辑（`sklearn_execution_time_predictor`）：

- **Prefill**：`prefill_chunk_size = batch.num_tokens[i]`；`kv_cache_size` 由 `request.num_processed_tokens` 按 `kv_cache_prediction_granularity`（默认 64）向上取整。
- **Decode**：`kv_cache_size = mean(num_processed_tokens)` 再按粒度上取整；`batch_size` = decode 请求数。

### 2.3 KV / prefix cache 命中

Frontier `Request.on_cache_hit(n)`（或等价地）设置：

- `num_processed_tokens = n`（已有 KV 长度）
- 仍保持 prefill 未完成（除非 `n >= num_prefill_tokens`）
- 本步 `Batch.num_tokens` 只含**剩余**要算的 token

因此命中后：线性类算子随 chunk 变小而变便宜；attention 仍用较大的 `kv_cache_size` 读已有上下文——**不是**简单的「命中 x% → 总算力减 x%」。

---

## 3. hybridsim analytical 怎么估时

1. `extract_batch_features(ScheduleBatch)` → `BatchFeatures`（phase、本步 prefill/decode token、`cached_decode_tokens`）。
2. `build_operator_dag` 按层展开形状原语（GEMM / Mem / 融合 Attn / Comm）。
3. `AnalyticAnalyzer`：Mem → bytes/bw；GEMM / 融合 Attn → Roofline；Comm → α-β；乘以全局 `duration_scale`。
4. 串行时长用 critical-path（可交叠时比 sum 更短）。

### 3.1 结构对齐

通信节点仍用 Frontier 风格短名（`mock/comm_names.py`）；计算节点用原语名（`gemm_qkv` / `fused_attn` 等）。`tests/test_op_workload_generator.py` 覆盖 DAG 名与依赖无环。旧 rf_op 公式对照在 `tests/rf_baseline/`。

### 3.2 数值对齐机制

解析模型与学习型 RF **不是同一模型**。当前对齐方式：

1. Roofline 使用 `DeviceConfig.compute_util` / `hbm_util`（默认 **0.6**）得到有效峰值。
2. `duration_scale=1.0`（**不再**对 Frontier 做全局 scale 拟合）。
3. 与 **非 dummy** Frontier RF（profiling CSV 训练）比 critical-path，相对误差 ≤ 5%（`MAX_REL_ERR = 0.05`）。

离线 `duration_scale` 拟合工具仍保留在本目录 `calibrator.py`（Mock 单测覆盖其数学），但不再作为 Frontier 对齐主路径。

### 3.3 KV 相关

| 场景 | 现状 |
|------|------|
| Prefill / Decode cache | **per-request** 列表；attention 对请求求和（FlashAttention-varlen，**不按 max pad**） |
| `cached_prefix_tokens` | prefill 请求 cached 之和（标量汇总） |
| `cached_decode_tokens` | **仅 decode** 请求 context 之和 |
| 线性 / FFN | 按本步 packed `num_tokens`（调度真实 token，无假 pad） |
| 设备效率 | `DeviceConfig.compute_util` / `hbm_util`（默认 0.6）作用于 Roofline 有效峰值 |

---

## 4. 对齐现状（实测）

测试：`analytic_workload_calibration.test_rf_calibration.TestFrontierRfNumericalAlignment`。  
明细表：`RF_ALIGNMENT.md`（测试通过时自动覆写）。

环境：`build_monolithic_context(..., enable_dummy_mode=False, model_name=llama2_7b_dense_example, device=h800)`。

实测（2026-08-12，非 dummy RF + util=0.6，**旧 rf_op 查表**）当时全部 ≤ 5%。

2026-08-30 形状原语重构后曾把 Q/K/V 拆成三个 GEMM（激活多读），`multi_prefill` rel_err ≈ 0.16。之后已改回与 vLLM `QKVParallelLinear` 一致的 **一次 packed GEMM**。5% 硬门仍只记录 `RF_ALIGNMENT.md`，不再 fail。

Mock 路径（`TestMockRfNumericalAlignment`）：用 `k * analytical` 作参考，可精确回收 `duration_scale`，验证标定数学正确。

**曾未覆盖 / 现已补齐部分：**

- mid-prefill / APC partial hit：analytical 已用 `cached_prefix_tokens` 计入 prefill attention；校准用例含 multi/single prefix-hit
- per-op 对 RF `execution_time_attr` 的对齐：**仍未做**
- `model_preset` 现为全 `duration_mode` 共用的 `ModelConfig` 源；与 Frontier `model_name` 的自动绑定：**仍未做**

---

## 5. 配置约定

- **`model_preset`**：模型形状、注意力变体、KV 字节公式的公用真源（op-level DAG + KV transfer；所有 `duration_mode` 均可挂载）。
- **`frontier_predictor`**：`batch_predictor=frontier` 的时长真源；应与 preset 指向同一模型族，但 RF 不读取 `ModelConfig` 字段做计时。

---

## 6. 相关路径

| 路径 | 角色 |
|------|------|
| `infer_workload_generator/batch_level/predictors/frontier.py` | ScheduleBatch → Frontier RF |
| `infer_workload_generator/op_level/generator.py` | features + mock DAG + analyzer |
| `op_level/mock/ops.py` / `fused.py` | 形状原语与融合核（`Op.apply` + `features()`） |
| `tests/rf_baseline/` | 冻结 rf_op 公式，仅用于块级对照测试 |
| `tests/analytic_workload_calibration/` | 本报告、标定工具与数值对齐测试 |
| `tools/profile_op_workload.py` | Op DAG Chrome Trace 导出 |

全量测试（仓库根目录）：

```bash
PYTHONPATH=src/python:tests:examples/frontier:${FRONTIER_ROOT:-/home/y_luchenda/Frontier} \
  python -m unittest discover -s tests -p 'test_*.py' -v
```
