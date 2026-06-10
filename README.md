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

## 网络问题

若 `git clone` 失败，可手动准备 `third_party/`，或配置镜像后重试。Git 操作通常比 HTTPS 下载 GitHub 首页更稳定；`codeload.github.com` 也可用于手动下载 zip。
