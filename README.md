# hybridsim

基于 [simcpp20](https://github.com/fschuetz04/simcpp20) 的离散事件仿真 **Actor** 平台，提供 C++ 与 Python 接口。

## 结构

```text
src/hybridsim/            C++ Actor / Engine 核心
src/python/binding/       pybind11 → hybridsim_py
src/python/hybridsim/     平台 Python 包（Simulation / Config / ActorBase，无 Frontier）
src/python/hybridsim_infer/  LLM 推理仿真（Cluster / Replica / KV / workload generator）
docs/                     推理仿真文档（见 docs/README.md）
examples/frontier/        用平台复现 Frontier 调度的实施例（见该目录 README）
tests/                    平台测试
```

## 推理仿真

`hybridsim_infer` 在本平台上搭了一套 LLM serving 仿真：请求生成 → 集群分发 → 实例调度 + KV → 算子 DAG → Engine 执行。

架构总览与分层文档入口：**[docs/README.md](docs/README.md)**（从 [docs/architecture.md](docs/architecture.md) 读起）。跑 demo 见 [examples/inference/README.md](examples/inference/README.md)。

## 依赖获取策略

构建时按以下顺序获取依赖：

```
third_party/simcpp20 或 third_party/pybind11 已存在？
  ├─ 是 → 直接使用
  └─ 否 → git clone 到 third_party/
           └─ 失败 → FetchContent 拉取（仅作兜底）
```

`third_party/` 已在 `.gitignore` 中。

## 环境要求

- CMake >= 3.14
- C++20 编译器（GCC >= 10 / Clang >= 14）
- Git（用于自动 clone 依赖）
- Python >= 3.10（可选，构建 `hybridsim_py` 时需要）

## 构建

### 推荐：pip 安装（Python + C++ 绑定）

```bash
pip install -e .
```

会通过 CMake 编译 `hybridsim_py`，并将 `hybridsim` 包装入当前环境。需要 CMake ≥ 3.14、C++20 编译器、Git、Python ≥ 3.10。

若系统 pip 过旧，先升级：`python3 -m pip install -U 'pip>=24' setuptools wheel`。若隔离构建拉取 cmake wheel 失败，可用：`python3 -m pip install --no-build-isolation -e .`。

安装后可直接：

```bash
python3 examples/actor_python_demo.py
```

### 测试

Python 测试已统一为标准库 `unittest`。一键跑平台 + Frontier 示例测试：

```bash
python3 run_tests.py
# 仅平台：
python3 run_tests.py --platform-only
# 等价：
python3 -m unittest discover -s tests -v
```

未安装 Frontier 时，Frontier 相关用例会自动 skip；对齐测试可用 `HYBRIDSIM_SKIP_FRONTIER_ALIGN=1` 跳过。

### 仅 CMake（C++ / 开发调试）

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure   # C++ + discover 平台 Python 测试
```

仅构建 C++（跳过 Python）：

```bash
cmake -B build -DHYBRIDSIM_BUILD_PYTHON=OFF
cmake --build build
```

未 `pip install` 时，可用 `SimulationConfig(build_dir=Path("build"))` 指向 CMake 产物目录。

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `HYBRIDSIM_BUILD_PYTHON` | ON | 构建 Python 模块 |
| `HYBRIDSIM_BUILD_TESTS` | ON | 注册 CTest 测试 |
| `HYBRIDSIM_BUILD_EXAMPLES` | ON | 构建示例程序 |

## Actor request / reply（同步请求-响应）

在保留 `send` / `on` 的前提下，可用 `request` 等待对方处理结果（DES 协程挂起，非线程阻塞）。

**C++**

```cpp
// 发送方（async handler 内）；delay 默认 0（立即投递）
target.send(SetMsg{1}, /*delay=*/1.0);
auto ev = target.request<Result>(QueryMsg{...}, /*delay=*/2.0);
Result r = co_await ev;

// 接收方（on 不变；可选 reply，否则 handler 结束自动空回复）
b.on<QueryMsg>([](actor &self, QueryMsg &msg) {
  self.reply(*self.current_request(), Result{...});
});
```

**Python**

```python
# 接收方：普通 def 即可
@on(QueryMsg)
def handle_query(self, _actor, msg):
    self.reply({"echo": msg.id})  # 或不调用 → 自动 None

# 发送方：须 async def 才能 await；delay 默认 0
@on(StartMsg)
async def handle_start(self, _actor, _msg):
    worker.send(SetMsg, delay=1.0, value=1)
    result = await self.request(worker, QueryMsg, delay=2.0, id=1)
```

## Python 平台使用

```python
from hybridsim import Simulation, SimulationConfig, ActorBase, on

sim = Simulation(SimulationConfig())
sim.register_messages([MyMsg])
sim.spawn_actor(MyActor)
sim.before_run = lambda: ...
sim.run()
sim.check_errors()
```

## Frontier 复现示例

Frontier 调度复现（定制 Config / Msg / Actor 注册到平台 `Simulation`）见：

**[examples/frontier/README.md](examples/frontier/README.md)**

## 网络问题

若 `git clone` 失败，可手动准备 `third_party/`，或配置镜像后重试。
