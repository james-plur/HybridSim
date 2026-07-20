#!/usr/bin/env python3
"""Run Frontier architecture cases on Frontier + hybridsim and compare profiles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "build"))

from hybridsim_scheduler.architecture_cases import (
    ArchitectureCase,
    build_cli_args,
    list_cases,
)
from hybridsim_scheduler.simulation_driver import load_config_from_cli_args, run_from_cli_args

from build_schedule_profile import build_frontier_profile, compare_profiles, write_profile


OUTPUT_ROOT = Path("/home/y_luchenda/hybridsim/outputs/architecture_compare")
FRONTIER_ROOT = Path("/home/y_luchenda/Frontier")


def run_frontier(case: ArchitectureCase, output_dir: Path) -> Path:
    from frontier.config.global_vars import reset_global_vars

    reset_global_vars()
    cli_args = build_cli_args(case, metrics_output_dir=output_dir, enable_trace=True)
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": f"{FRONTIER_ROOT}:{__import__('os').environ.get('PYTHONPATH', '')}",
        "WANDB_DISABLED": "true",
        "VIDUR_DISABLE_WANDB": "1",
    }
    cmd = [sys.executable, "-m", "frontier.main", *cli_args]
    subprocess.run(cmd, cwd=FRONTIER_ROOT, env=env, check=True)
    config = load_config_from_cli_args(cli_args)
    return Path(config.metrics_config.output_dir)


def run_hybridsim(case: ArchitectureCase, output_dir: Path) -> Path:
    from frontier.config.global_vars import reset_global_vars

    reset_global_vars()
    cli_args = build_cli_args(case, metrics_output_dir=output_dir, enable_trace=False)
    import os

    previous_cwd = os.getcwd()
    try:
        os.chdir(FRONTIER_ROOT)
        config = load_config_from_cli_args(cli_args)
        run_from_cli_args(
            cli_args,
            build_dir=ROOT / "build",
            trace_output_dir=Path(config.metrics_config.output_dir),
        )
        return Path(config.metrics_config.output_dir)
    finally:
        os.chdir(previous_cwd)


def run_case(case: ArchitectureCase) -> dict:
    case_key = case.case_id.replace("/", "__")
    frontier_dir = OUTPUT_ROOT / "frontier" / case_key
    hybridsim_dir = OUTPUT_ROOT / "hybridsim" / case_key
    frontier_dir.mkdir(parents=True, exist_ok=True)
    hybridsim_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {case.case_id} :: Frontier ===")
    frontier_run_dir = run_frontier(case, frontier_dir)
    frontier_profile = write_profile(frontier_run_dir, "frontier")

    print(f"=== {case.case_id} :: hybridsim ===")
    hybridsim_run_dir = run_hybridsim(case, hybridsim_dir)
    hybridsim_profile = hybridsim_run_dir / "inference_profile.json"

    comparison = compare_profiles(frontier_profile, hybridsim_profile)
    comparison_path = OUTPUT_ROOT / "comparisons" / f"{case_key}.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    status = "PASS" if comparison["batch_mismatches"] == 0 else "DIFF"
    print(
        f"=== {case.case_id} :: {status} "
        f"(batch_match_rate={comparison['match_rate']:.1%}, "
        f"batch_mismatches={comparison['batch_mismatches']}/"
        f"{comparison['frontier_batch_count']})"
    )
    return {
        "case": case.case_id,
        "status": status,
        "frontier_run_dir": str(frontier_run_dir),
        "hybridsim_run_dir": str(hybridsim_run_dir),
        "comparison": comparison,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-filter", default="", help="Substring filter for case id")
    parser.add_argument("--arch", choices=["co-location", "pdd", ""], default="")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    cases = list_cases()
    if args.arch:
        cases = [case for case in cases if case.arch == args.arch]
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.case_id]
    if args.list:
        for case in cases:
            print(case.case_id)
        return
    if not cases:
        raise SystemExit("No cases matched filter")

    results = []
    for case in cases:
        try:
            results.append(run_case(case))
        except Exception as exc:
            print(f"=== {case.case_id} :: FAILED ({exc})")
            results.append(
                {
                    "case": case.case_id,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(1 for item in results if item.get("status") == "PASS")
    failed = sum(1 for item in results if item.get("status") == "FAILED")
    differ = sum(1 for item in results if item.get("status") == "DIFF")
    print(
        f"\nCompleted {len(results)} case(s): "
        f"{passed} passed, {differ} differ, {failed} failed."
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
