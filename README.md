# hybridsim

基于 [simcpp20](https://github.com/fschuetz04/simcpp20) 的离散事件仿真 Actor 框架，提供 C++ 与 Python 接口。

## 依赖获取策略

构建时按以下顺序获取依赖：

```
third_party/simcpp20 或 third_party/pybind11 已存在？
  ├─ 是 → 直接使用
  └─ 否 → git clone 到 third_party/
           └─ 失败 → FetchContent 拉取（仅作兜底）
```

`third_party/` 已在 `.gitignore` 中，clone 后的依赖不会进入版本库。

## 环境要求

- CMake >= 3.14
- C++20 编译器（GCC >= 10 / Clang >= 14）
- Git（用于自动 clone 依赖）
- Python >= 3.8（可选，构建 `hybridsim_py` 时需要）

## 构建

```bash
# 完整构建（自动 clone 缺失依赖）
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

若已有本地依赖：

```bash
mkdir -p third_party
git clone --depth 1 --branch main https://github.com/fschuetz04/simcpp20.git third_party/simcpp20
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git third_party/pybind11
cmake -B build && cmake --build build
```

仅构建 C++（跳过 Python）：

```bash
cmake -B build -DHYBRIDSIM_BUILD_PYTHON=OFF
cmake --build build
```

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `HYBRIDSIM_BUILD_PYTHON` | ON | 构建 Python 模块 |
| `HYBRIDSIM_BUILD_TESTS` | ON | 注册 CTest 测试 |
| `HYBRIDSIM_BUILD_EXAMPLES` | ON | 构建示例程序 |
| `HYBRIDSIM_SIMCPP20_REPO` / `TAG` | GitHub / `main` | simcpp20 仓库地址与版本 |
| `HYBRIDSIM_PYBIND11_REPO` / `TAG` | GitHub / `v2.13.6` | pybind11 仓库地址与版本 |

## Python 使用

```bash
export PYTHONPATH=/path/to/hybridsim/build
python3 examples/actor_python_demo.py
```

## Scheduler + Frontier

MONOLITHIC / PDD 调度集成通过 [Frontier](https://github.com/NetX-lab/Frontier) 提供 batching 与 predictor 逻辑；hybridsim 通过 Actor 桥接层（`src/python/hybridsim_scheduler/`）驱动仿真时钟。

### 安装 Frontier

```bash
pip install /path/to/Frontier --no-build-isolation
```

Frontier 包名为 `frontier-simulator`（见 Frontier 仓库的 `pyproject.toml`），会安装 `fasteners`、`scikit-learn` 等依赖。

### 运行调度测试与示例

```bash
cmake -B build -DHYBRIDSIM_BUILD_EXAMPLES=OFF
cmake --build build

export PYTHONPATH=/path/to/Frontier:/path/to/hybridsim/build:/path/to/hybridsim/src/python
python3 tests/test_scheduler_monolithic.py
python3 examples/scheduler_monolithic_demo.py
```

### 复现 Frontier architecture 示例并对比 Chrome profile

对 `Frontier/examples/architecture` 下 20 个 co-location + PDD 用例，分别运行 Frontier 原生仿真与 hybridsim 复现，并生成 `chrome://tracing` 可读的 `inference_profile.json`：

```bash
export PYTHONPATH=/path/to/Frontier:/path/to/hybridsim/build:/path/to/hybridsim/src/python
python3 examples/run_architecture_matrix.py
```

产物目录：`outputs/architecture_compare/`

- `frontier/<case>/.../inference_profile.json` — Frontier 调度时间线
- `hybridsim/<case>/.../inference_profile.json` — hybridsim 调度时间线
- `comparisons/<case>.json` — batch 级调度窗口对比结果

可用 `--case-filter dense` 或 `--arch pdd` 过滤子集。

也可安装 hybridsim 调度包（可选依赖声明 Frontier）：

```bash
pip install -e ".[scheduler]" /path/to/hybridsim
```

## 网络问题

若 `git clone` 失败，可手动准备 `third_party/`，或配置镜像后重试。Git 操作通常比 HTTPS 下载 GitHub 首页更稳定；`codeload.github.com` 也可用于手动下载 zip。
