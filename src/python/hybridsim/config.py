"""Platform simulation configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypeVar

T = TypeVar("T", bound="SimulationConfig")


@dataclass
class SimulationConfig:
    """Base hybridsim simulation configuration.

    Holds options that apply to every simulation kind. Scenario-specific fields
    belong on subclasses. Subclasses may override ``from_cli_args`` to parse
    additional flags (optionally calling ``parse_common_cli_args`` first).
    """

    #: Optional CMake build dir containing ``hybridsim_py`` when not installed
    #: via ``pip install -e .``. Prefer installing the package instead.
    build_dir: Optional[Path] = None
    #: If set, schedule traces are written under this directory.
    trace_output_dir: Optional[Path] = None

    @classmethod
    def parse_common_cli_args(
        cls, argv: Sequence[str]
    ) -> tuple[dict, list[str]]:
        """Parse platform-common flags; return (kwargs, remaining argv)."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--build_dir",
            type=Path,
            default=None,
            help="Optional CMake build dir with hybridsim_py (prefer pip install -e .)",
        )
        parser.add_argument("--trace_output_dir", type=Path, default=None)
        args, remaining = parser.parse_known_args(list(argv))
        kwargs = {
            "build_dir": args.build_dir,
            "trace_output_dir": args.trace_output_dir,
        }
        return kwargs, remaining

    @classmethod
    def from_cli_args(cls: type[T], argv: Sequence[str] | None = None) -> T:
        """Build a config from CLI-style arguments (platform-common flags only)."""
        import sys

        raw = list(sys.argv[1:] if argv is None else argv)
        kwargs, _remaining = cls.parse_common_cli_args(raw)
        return cls(**kwargs)
