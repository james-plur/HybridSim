"""Build and install hybridsim (Python package + C++ pybind11 extension).

Usage:
  pip install -e .          # editable: compile hybridsim_py, install hybridsim
  pip install .             # regular install
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        extdir.mkdir(parents=True, exist_ok=True)

        build_temp = Path(self.build_temp).resolve() / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DHYBRIDSIM_BUILD_PYTHON=ON",
            "-DHYBRIDSIM_BUILD_TESTS=OFF",
            "-DHYBRIDSIM_BUILD_EXAMPLES=OFF",
        ]
        for config_name in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel"):
            cmake_args.append(
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config_name.upper()}={extdir}"
            )

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                arg for arg in os.environ["CMAKE_ARGS"].split(" ") if arg
            ]

        build_args: list[str] = ["--target", "hybridsim_py"]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            jobs = getattr(self, "parallel", None) or os.cpu_count() or 2
            build_args += ["--parallel", str(jobs)]

        configure_cmd = [
            "cmake",
            "-S",
            ext.sourcedir,
            "-B",
            str(build_temp),
        ]
        generator = os.environ.get("CMAKE_GENERATOR")
        if generator:
            configure_cmd += ["-G", generator]
        configure_cmd += cmake_args

        subprocess.check_call(configure_cmd)
        subprocess.check_call(["cmake", "--build", str(build_temp), *build_args])

        expected = Path(self.get_ext_fullpath(ext.name))
        if expected.exists():
            return

        candidates = list(extdir.glob(f"{ext.name}*.so")) + list(
            extdir.glob(f"{ext.name}*.pyd")
        )
        if not candidates:
            raise RuntimeError(
                f"CMake built hybridsim_py but {expected} was not found "
                f"(searched {extdir})"
            )
        match = next(
            (c for c in candidates if c.suffix == expected.suffix),
            candidates[0],
        )
        if match.resolve() != expected.resolve():
            expected.parent.mkdir(parents=True, exist_ok=True)
            match.replace(expected)


_HERE = Path(__file__).resolve().parent
_README = _HERE / "README.md"

setup(
    name="hybridsim",
    version="0.1.0",
    description=(
        "Actor-based discrete-event simulation platform "
        "(Python bindings + infrastructure)."
    ),
    long_description=_README.read_text(encoding="utf-8") if _README.is_file() else "",
    long_description_content_type="text/markdown",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python", include=["hybridsim*"]),
    ext_modules=[CMakeExtension("hybridsim_py")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    zip_safe=False,
)
