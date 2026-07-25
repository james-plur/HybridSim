#!/usr/bin/env python3
"""Run a schedule-alignment case: hybridsim (+ optional vLLM) ledgers and compare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m schedule_alignment.run_case` with PYTHONPATH=tests:.
# and direct script execution from repo root.
_REPO = Path(__file__).resolve().parents[2]
_TESTS = _REPO / "tests"
_PY = _REPO / "src" / "python"
for _p in (_TESTS, _REPO, _PY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schedule_alignment.case_loader import list_cases, load_case
from schedule_alignment.compare import compare_ledgers
from schedule_alignment.hybridsim_schedule_driver import run_hybridsim_schedule
from schedule_alignment.schema import write_ledger
from schedule_alignment.vllm_schedule_driver import (
    ensure_vllm_path,
    run_vllm_schedule,
    vllm_available,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=False,
        help="Case name (under cases/) or path to JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for ledger outputs (default: cases/)",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Only run hybridsim driver",
    )
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="Write hybridsim ledger as cases/<name>.expected.ledger.jsonl",
    )
    args = parser.parse_args(argv)

    if args.list or not args.case:
        names = list_cases()
        print("cases:", ", ".join(names) if names else "(none)")
        if not args.case:
            return 0 if args.list else 2

    case = load_case(args.case)
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    hs_records = run_hybridsim_schedule(case)
    hs_path = out_dir / f"{case.name}.hybridsim.ledger.jsonl"
    write_ledger(hs_path, hs_records)
    print(f"wrote {hs_path} ({len(hs_records)} steps)")

    if args.write_expected:
        exp_path = out_dir / f"{case.name}.expected.ledger.jsonl"
        write_ledger(exp_path, hs_records)
        print(f"wrote {exp_path}")

    exp_path = out_dir / f"{case.name}.expected.ledger.jsonl"
    if exp_path.exists() and not args.write_expected:
        from schedule_alignment.schema import read_ledger

        report = compare_ledgers(
            read_ledger(exp_path),
            hs_records,
            left_name="expected",
            right_name="hybridsim",
        )
        print(report.summary())
        if not report.equal:
            return 1

    if not args.skip_vllm:
        ensure_vllm_path()
        if not vllm_available():
            print(
                "ERROR: vLLM/torch unavailable. Install torch + set VLLM_ROOT, "
                "or pass --skip-vllm for hybridsim-only.",
                file=sys.stderr,
            )
            return 2
        try:
            vllm_records = run_vllm_schedule(case)
        except Exception as exc:
            print(f"vLLM driver failed: {exc}", file=sys.stderr)
            return 1
        vllm_path = out_dir / f"{case.name}.vllm.ledger.jsonl"
        write_ledger(vllm_path, vllm_records)
        print(f"wrote {vllm_path} ({len(vllm_records)} steps)")
        report = compare_ledgers(
            vllm_records,
            hs_records,
            left_name="vllm",
            right_name="hybridsim",
        )
        print(report.summary())
        if not report.equal:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
