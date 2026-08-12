"""One-command local release audit before uploading Stage 7.4 to AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "evaluate_stage7.py", "scripts/run_stage74_calibration_autodl.sh", "scripts/run_stage7_autodl.sh",
    "scripts/preflight_stage74_autodl.py", "scripts/validate_stage74_calibration.py",
    "scripts/generate_final_report.py", "scripts/generate_stage7_manual_review.py",
    "experiments/stage74_fix_regression.jsonl", "experiments/eval_queries_final_labeled.jsonl",
    "experiments/eval_queries_final_random.jsonl", "artifacts/stage7/before_fix/before_fix_manifest.json",
)


def audit(*, run_tests: bool = False) -> dict:
    checks = []
    for relative in REQUIRED:
        path = ROOT / relative
        checks.append({"name": relative, "passed": path.is_file(), "detail": str(path)})
    verify = subprocess.run(
        [sys.executable, "scripts/archive_stage74_before_fix.py", "--verify-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    checks.append({"name": "before_fix_integrity", "passed": verify.returncode == 0, "detail": verify.stdout[-500:]})
    if run_tests:
        test = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--basetemp", "artifacts/pytest_stage74_release_audit"],
            cwd=ROOT, capture_output=True, text=True,
        )
        checks.append({"name": "full_unit_tests", "passed": test.returncode == 0, "detail": (test.stdout + test.stderr)[-1000:]})
    hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in REQUIRED if (ROOT / relative).is_file()
    }
    return {"passed": all(item["passed"] for item in checks), "checks": checks, "release_hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage7/stage74_release_audit.json")
    args = parser.parse_args()
    report = audit(run_tests=args.run_tests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "check_count": len(report["checks"]), "out": str(args.out)}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
