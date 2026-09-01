# workload generator

> **本层位置**：replica 内部的第三步（`schedule` → `kv` → `workload generator`），把已调度的 `ScheduleBatch` 变成 Engine 能跑的算子 DAG 与时长。全景见 [`architecture.md`](architecture.md)，DAG 如何被执行见 [`engine.md`](engine.md)。

一次推理迭代对应一个计算图 op DAG。构图贴近 torch：用 mock `Module.forward` 展开图，叶子是**形状驱动的原语**，再交给 analyzer 估时。

## 分层

- **infer / batch_level**：按 `ScheduleBatch` 预测整 batch 时长（fixed / token-proportional / Frontier RF）。
- **infer / op_level**：mock 构图 + analytic 分析。
- **kv**：独立 KV 传输 workload，与 infer 共用 `model_presets` / `ModelConfig`。

`BatchFeatures` 只用于 `Module.forward`（拼 packed 激活、为 varlen 构造每条请求的 Q/K 形状），**不写入叶子 Op**。

## 叶子原语

| 类 | 记录 | 代价 |
|----|------|------|
| `MemOp` | 若干形状 + `dtype_bytes` | `bytes = dtype * sum(numel)`，`flops = 0`；`T = mem_scale * bytes / bw` |
| `GemmOp` | `A[M,K]`、`B[K,N]` | `flops = 2MNK`，`bytes = dtype*(MK+KN+MN)`；Roofline |
| `FusedAttnOp` / `FusedMlaAttnOp` | Q/K/V（或 latent）形状 + `kernel=prefill\|decode` | 独立融合公式 |
| `CommOp` | payload 形状 + collective + `num_ranks` | α-β；`ranks<=1` 不建节点 |

并行只在构图时通过 `Shape.split` 和是否插入 Comm 体现。`Op.name` 仅调试（如 `L3.gemm_qkv`）。

与旧 Frontier rf_op 公式的块级对照在 `tests/rf_baseline/`，不进入主路径。

## Mock 构图

- `Module`：子模块自动注册
- `Shape`：权重；`clone().split(dim, tp)` 做 TP 切分，结果进入 GEMM
- `Tensor`：激活形状 + `dtype_bytes` + 生产者下标
- `Graph` / `CommContext`：contextvar 记录节点

```python
class Attention(Module):
    def forward(self, x, *, layer_id, batch):
        qkv = GemmOp.apply(x, qkv_w, name=f"L{layer_id}.gemm_qkv")  # packed Q|K|V
        # ...
        for chunk, cached in batch.iter_prefill_attn_pairs():
            FusedAttnOp.apply(
                qkv,
                q_shape=(chunk, n_q, d),
                k_shape=(cached + chunk, n_kv, d),
                v_shape=(cached + chunk, n_kv, d),
                kernel="prefill",
                name=f"L{layer_id}.fused_attn",
                dtype_bytes=x.dtype_bytes,
            )
        return GemmOp.apply(deps, (s, n_q * d), o_w, name=f"L{layer_id}.gemm_o", out_shape=(s, h))
```

`first_k_dense_replace` 按全局 `layer_id` 切换 dense FFN / MoE。DSA v1 = MLA 融合核 + indexer `MemOp`，无稀疏 attn 核。

## Analyzer

估时：`lower_op` 只读 `op.features()` + `op.kind`，不再按融合子类分支。**计算**和**通信**可以配不同的 analyzer：

- `AnalyticAnalyzer`（`compute_analyzer=analytic`）：Mem → `mem_scale * bytes / bw`；GEMM / FUSED → Roofline；若未挂 comm analyzer，Comm → α-β TimeoutKernel。
- `RingCommAnalyzer`（`comm_analyzer=ring`）：只处理 `CommOp`，展开成 Put/Wait 等；计算 op 仍走 analytic。

`OpLevelConfig.compute_analyzer` / `comm_analyzer` 分开选择。开启 [`network_sim`](network.md) 且 comm 仍为 analytic 时，自动改用 ring。新融合核只需加 `FusedOp` 子类。
