"""Alignment smoke: compare one Frontier architecture case vs hybridsim.

Skip with: HYBRIDSIM_SKIP_FRONTIER_ALIGN=1
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

FRONTIER_EXAMPLE = Path(__file__).resolve().parents[1]
if str(FRONTIER_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(FRONTIER_EXAMPLE))

try:
    import frontier  # noqa: F401

    _FRONTIER_AVAILABLE = True
except ImportError:
    _FRONTIER_AVAILABLE = False

_SKIP_ALIGN = os.environ.get("HYBRIDSIM_SKIP_FRONTIER_ALIGN", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


@unittest.skipUnless(_FRONTIER_AVAILABLE, "Frontier not installed (pip install -e $FRONTIER_ROOT)")
@unittest.skipIf(_SKIP_ALIGN, "HYBRIDSIM_SKIP_FRONTIER_ALIGN set")
class ArchitectureAlignTests(unittest.TestCase):
    def test_architecture_align_dense_offline(self) -> None:
        from frontier_bridge.architecture_cases import ArchitectureCase
        from run_architecture_matrix import run_case

        case = ArchitectureCase("co-location", "offline/dense_model_basic.sh")
        result = run_case(case)
        self.assertNotEqual(result["status"], "FAILED", result)
        self.assertIn("comparison", result)


if __name__ == "__main__":
    unittest.main()
