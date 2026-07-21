# hybridsim

基于 [simcpp20](https://github.com/fschuetz04/simcpp20) 的离散事件仿真 **Actor** 平台，提供 C++ 与 Python 接口。

## 结构

```text
src/hybridsim/          C++ Actor / Engine 核心
src/python/binding/     pybind11 → hybridsim_py
src/python/hybridsim/   平台 Python 包（Simulation / Config / ActorBase，无 Frontier）
examples/frontier/      用平台复现 Frontier 调度的实施例（见该目录 README）
tests/                  平台测试
```

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
