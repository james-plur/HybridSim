#!/usr/bin/env python3
"""One-shot unittest runner for hybridsim.

Discovers and runs:
  - platform tests under tests/
  - Frontier example tests under examples/frontier/tests/

Usage:
  python run_tests.py
  python run_tests.py --platform-only
  python run_tests.py -q
  HYBRIDSIM_SKIP_FRONTIER_ALIGN=1 python run_tests.py
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTIER_EXAMPLE = ROOT / "examples" / "frontier"


def _discover(start: Path, pattern: str = "test_*.py") -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover(str(start), pattern=pattern, top_level_dir=str(start))


def build_suite(*, platform: bool = True, frontier: bool = True) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    if platform:
        suite.addTests(_discover(ROOT / "tests"))
    if frontier:
        if str(FRONTIER_EXAMPLE) not in sys.path:
            sys.path.insert(0, str(FRONTIER_EXAMPLE))
        suite.addTests(_discover(FRONTIER_EXAMPLE / "tests"))
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hybridsim unittest suites")
    parser.add_argument(
        "--platform-only",
        action="store_true",
        help="Only run tests/ (skip examples/frontier/tests)",
    )
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help="Only run examples/frontier/tests",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    parser.add_argument(
        "-f",
        "--failfast",
        action="store_true",
        help="Stop on first failure",
    )
    args = parser.parse_args(argv)

    if args.platform_only and args.frontier_only:
        parser.error("use only one of --platform-only / --frontier-only")

    platform = not args.frontier_only
    frontier = not args.platform_only
    suite = build_suite(platform=platform, frontier=frontier)
    verbosity = 1 if args.quiet else 2
    result = unittest.TextTestRunner(
        verbosity=verbosity, failfast=args.failfast
    ).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
