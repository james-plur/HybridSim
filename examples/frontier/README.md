# Frontier + hybridsim 实施例

本目录演示如何用 **hybridsim 平台**（Actor / Msg / `Simulation` / `SimulationConfig`）挂载 [Frontier](https://github.com/NetX-lab/Frontier) 的调度与预测逻辑，并与 Frontier 原生仿真对齐。

## 设计

```text
hybridsim.Simulation          # 平台：消息注册、actor 生命周期、run/stop/check_errors
        ▲
        │ 注册定制 Msg / Actor / Config
frontier_bridge
  ArchitectureConfig / MonolithicConfig
  ClusterSchedulerActor / ReplicaSchedulerActor
  RequestArrivalMsg / …
```

- 平台包：`src/python/hybridsim/`（**不**依赖 Frontier）
- 本示例：`examples/frontier/frontier_bridge/` 在 `Simulation` 上注册业务组件

## 安装

在仓库根目录安装 hybridsim（会编译 C++ 绑定 `hybridsim_py`，并安装 Python 包）：

```bash
cd /path/to/hybridsim
pip install -e .
```

需要：CMake ≥ 3.14、C++20 编译器、Git（拉取 simcpp20 / pybind11）、Python ≥ 3.10。

若系统自带的 pip 过旧（会退回 `setup.py develop` 并报权限错误），先升级再装：

```bash
python3 -m pip install -U 'pip>=24' setuptools wheel
python3 -m pip install -e .
```

若隔离构建下载 cmake wheel 失败，可用本机已有的 CMake/Ninja：

```bash
python3 -m pip install --no-build-isolation -e .
```

再安装 Frontier：

```bash
export FRONTIER_ROOT=/path/to/Frontier   # 可选；默认 /home/y_luchenda/Frontier
pip install -e "$FRONTIER_ROOT"
```

运行本示例时，把 `examples/frontier` 放进 `PYTHONPATH`（以便 `import frontier_bridge`）：

```bash
export PYTHONPATH="$PWD/examples/frontier${PYTHONPATH:+:$PYTHONPATH}"
```

> 仅用裸 CMake 构建、未 `pip install` 时，可继续 `cmake -B build && cmake --build build`，并在 `SimulationConfig(build_dir=...)` 中指向 `build/`。推荐优先使用 `pip install -e .`。

## 快速冒烟（MONOLITHIC）

```bash
python3 examples/frontier/scheduler_monolithic_demo.py
```

一键跑全部 Python 测试（平台 + 本目录；基于 `unittest`）：

```bash
python3 run_tests.py
# 仅 Frontier：
python3 run_tests.py --frontier-only
python3 -m unittest discover -s examples/frontier/tests -v
```

## 架构矩阵 / 对齐

与 Frontier 原生跑同一套 architecture case，对比 schedule profile：

```bash
python3 examples/frontier/run_architecture_matrix.py --list
python3 examples/frontier/run_architecture_matrix.py --case-filter dense_model_basic
```

轻量对齐测试（默认跑 `co-location/offline/dense_model_basic.sh`，含在 `run_tests.py` 中）：

```bash
# 跳过对齐用例：
HYBRIDSIM_SKIP_FRONTIER_ALIGN=1 python3 run_tests.py --frontier-only
```

产物默认写到仓库根目录 `outputs/architecture_compare/`。

## 入口 API

```python
from frontier_bridge import MonolithicConfig, ReplicaSchedulerKind, build_frontier_simulation

sim = build_frontier_simulation(
    MonolithicConfig(replica_scheduler_kind=ReplicaSchedulerKind.VLLM_V1)
)
sim.inject_requests([sim.add_request(arrived_at=0.0, num_prefill_tokens=4, num_decode_tokens=2)])
sim.run()
sim.check_errors()
```

CLI architecture：

```python
from frontier_bridge import run_from_cli_args
run_from_cli_args(cli_args, trace_output_dir=...)
```
