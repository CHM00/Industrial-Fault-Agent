"""Small deterministic P0 gate; extend with expert-labelled diagnosis cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety import assess_fault_safety


def main() -> int:
    cases = json.loads((Path(__file__).parent / "golden_safety_cases.json").read_text("utf-8"))
    failures = []
    for case in cases:
        actual = assess_fault_safety(case["fault"])
        checks = {
            "expected_level": actual["risk_level"] == case["expected_level"],
            "requires_approval": actual["requires_expert_approval"] == case["requires_approval"],
        }
        if "prohibited" in case:
            checks["prohibited"] = actual["prohibited"] == case["prohibited"]
        if not all(checks.values()):
            failures.append({"id": case["id"], "checks": checks, "actual": actual})
    print(json.dumps({"cases": len(cases), "passed": len(cases) - len(failures), "failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
