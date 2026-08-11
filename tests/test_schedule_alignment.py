"""Unit tests for schedule alignment cases (hybridsim ↔ expected / vLLM)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "tests"
_PY = _REPO / "src" / "python"
for _p in (_TESTS, _REPO, _PY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schedule_alignment.case_loader import list_cases, load_case
from schedule_alignment.compare import compare_ledgers
from schedule_alignment.hybridsim_schedule_driver import run_hybridsim_schedule
from schedule_alignment.schema import read_ledger
from schedule_alignment.vllm_schedule_driver import run_vllm_schedule, vllm_available

_CASES_DIR = _TESTS / "schedule_alignment" / "cases"


def _compare_kv_for(case) -> bool:
    return bool((case.scheduler or {}).get("enable_prefix_caching", False))


class TestScheduleAlignmentExpected(unittest.TestCase):
    """Each case ledger must match the checked-in expected golden."""

    def test_all_cases_match_expected_ledgers(self) -> None:
        names = list_cases()
        self.assertTrue(names, msg="no schedule alignment cases found")
        for name in names:
            with self.subTest(case=name):
                case = load_case(name)
                got = run_hybridsim_schedule(case)
                exp_path = _CASES_DIR / f"{name}.expected.ledger.jsonl"
                self.assertTrue(exp_path.exists(), msg=f"missing {exp_path}")
                report = compare_ledgers(
                    read_ledger(exp_path),
                    got,
                    left_name="expected",
                    right_name="hybridsim",
                    compare_kv=_compare_kv_for(case),
                )
                self.assertTrue(report.equal, msg=report.summary())


class TestScheduleAlignmentVllm(unittest.TestCase):
    """Each case hybridsim ledger must match offline vLLM Scheduler."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PYTHONHASHSEED", "0")

    def test_vllm_available(self) -> None:
        self.assertTrue(
            vllm_available(),
            msg="vLLM/torch must be installed for schedule alignment tests",
        )

    def test_all_cases_match_vllm_ledgers(self) -> None:
        self.assertTrue(
            vllm_available(),
            msg="vLLM/torch required for schedule alignment vLLM compare",
        )
        names = list_cases()
        self.assertTrue(names, msg="no schedule alignment cases found")
        for name in names:
            with self.subTest(case=name):
                case = load_case(name)
                hs = run_hybridsim_schedule(case)
                vllm = run_vllm_schedule(case)
                report = compare_ledgers(
                    vllm,
                    hs,
                    left_name="vllm",
                    right_name="hybridsim",
                    compare_kv=_compare_kv_for(case),
                )
                self.assertTrue(report.equal, msg=report.summary())


if __name__ == "__main__":
    unittest.main()
