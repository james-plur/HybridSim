# Network 流级仿真

> **本层位置**：Engine 之下的可选 fabric。默认关闭时通信仍是 α-β TimeoutKernel（NO_NETWORK）。开启后，计算仍走 Roofline `TimeoutKernel`，集体通信被 `RingCommAnalyzer` 拆成 Put/Signal/Wait/Get，在 C++ 数据面上做流级共享。拓扑连线和路由表在 **Python** 里初始化，便于扩展。全景见 [`architecture.md`](architecture.md)。

KV / Store 数据面**不**走这条网络，继续用 `kv_workload` 的 α-β。

## 何时启用

```python
from hybridsim_infer import (
    InferenceConfig,
    InferWorkloadConfig,
    NetworkSimConfig,
    ParallelConfig,
    build_inference_simulation,
)

cfg = InferenceConfig(
    infer_workload=InferWorkloadConfig(mode="op_level"),
    network_sim=NetworkSimConfig(enabled=True, topology="fattree", layers=1),
)
cfg.infer_workload.op.parallel = ParallelConfig(tp_size=2)
# 计算 / 通信 analyzer 可分开配：默认 compute=analytic；开启 network 后 comm 自动用 ring
cfg.infer_workload.op.compute_analyzer = "analytic"
cfg.infer_workload.op.comm_analyzer = "ring"
infra = build_inference_simulation(cfg)
```

`network_sim.enabled=False`（默认）时：每个 replica 仍是 1 个 Engine，CommOp 估时走 [`NetworkConfig`](../src/python/hybridsim_infer/workload_generators/configs.py) α-β（`comm_analyzer=analytic`）。

## Python 拓扑与路由

C++ 只提供节点、双向链路和路由表槽位。组网在 [`hybridsim.network`](../src/python/hybridsim/network/)：

| 原语 | 作用 |
|------|------|
| `Network.add_adapter(replica, rank, port_num=2)` | 端点（port 0 = host / Engine） |
| `Network.add_switch(port_num)` | 交换机 |
| `Network.link(a, a_port, b, b_port, bw, delay)` | `A.out[i] ↔ B.in[j]` |
| `Network.set_nexthops(node, dst_replica, dst_rank, ports)` | 等价下一跳出端口 |
| `Network.downstream(node, out_port)` | 查对端 `(node_id, in_port)` |

```python
from hybridsim.network import (
    FatTreeTopology,
    ShortestPathRouting,
    Topology,
    assemble_network,
    register_topology,
)

# 内置：fattree（1/2 层）+ shortest_path（全部最短下一跳）
net = assemble_network(sim.hs_sim, [(0, 0), (0, 1)], topology="fattree", layers=1)

# 自定义连线：注册 Topology 子类即可
@register_topology("line")
class LineTopology(Topology):
    def wire(self, net, addrs, *, bandwidth_bps, delay_s):
        a = net.add_adapter(*addrs[0], port_num=2)
        b = net.add_adapter(*addrs[1], port_num=2)
        net.link(a, 1, b, 1, bandwidth_bps, delay_s)
```

同样可以 `@register_routing("...")` 换路由算法。`network_sim.topology` / `network_sim.routing` 填注册名。

内置 FatTree：`layers=1` 单 switch；`layers=2` leaf–spine；`num_leaf` 等为 0 时按端点数自动推导。

## 数据面（C++）

实现在 [`src/hybridsim/network/`](../src/hybridsim/network/)。

- **NetworkAdapter**：端侧（网卡/IOD）。地址 `replica_id:rank`。port 0 接 Engine，port 1+ 接交换机。
- **NetworkSwitch**：交换机。`forward` 查路由表。
- **InPort / OutPort**：C++ Actor。
- **流消息**：`FlowArriveMsg` / `FlowUpdateMsg` / `FlowEndMsg`（带 `version`，带宽变化后旧 End 作废）。Get 的 fetch 控制流：目的 Adapter 收完后自动回一条数据流。

带宽策略：`max_min`、`ingress_proportional`、`priority_then_maxmin`。  
等价路径负载均衡（查表之后）：`ecmp_hash`、`random`、`least_loaded`。

## Comm kernel

| type | 名字 | 行为 |
|------|------|------|
| 0 | TimeoutKernel | 计算 / α-β 路径 |
| 1 | Put | inject 后立刻返回 |
| 2 | Signal | 同 Put，默认 64B |
| 3 | Wait | `co_await` 目的端 FlowEnd |
| 4 | Get | 发 fetch，再 wait 回包 |

Engine：`install_network(network, replica_id, rank)`。Wait 在**收完**时唤醒，不是 Arrive。

## Comm analyzer（Python，可选）

`RingCommAnalyzer` 是一种 **OpAnalyzer**：只把 `CommOp` 转成 per-rank primitive；计算 op 仍走 `AnalyticAnalyzer`（Roofline TimeoutKernel）。两者在 `OpLevelConfig.compute_analyzer` / `comm_analyzer` 上分开配置。

- AllReduce：ring，`2(n-1)` 步；同一步 Put/Wait 无互依赖
- ReduceScatter / AllGather：各 `n-1` 步
- P2P：rank 0 Put、rank 1 Wait
- EP dispatch/combine：all-to-all 对向 Put/Wait

`conn_id` 由 `(op_index, step, sender_rank[, extra])` 编码，双方一致。

`OpLevelWorkloadGenerator` 在挂了 comm analyzer 时产出：

```python
{"workload_id": wid, "per_rank": {rank: {"kernels": [...]}, ...}}
```

Replica 对每个 rank 一个 Engine，**全部** `WorkloadDoneMsg` 到齐才 `BatchEndMsg`。`ranks_per_replica = max(attn_tp, moe_tp, ep_size)`，可用 `network_sim.ranks_per_replica` 覆盖。

## 配置

见 [`inference_config.md`](inference_config.md) 的 `network_sim` 组。与 `infer_workload.op.network`（α-β）不是同一条链路。
