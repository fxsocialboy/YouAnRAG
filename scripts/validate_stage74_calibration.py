"""Validate the 51-row after-fix calibration before expensive full runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/stage7/after_fix/regression_eval.json"
DEFAULT_REPORT = ROOT / "artifacts/stage7/after_fix/calibration_acceptance.json"


FALSE_FALLBACK_IDS = {"final_006", "final_014", "final_019", "final_020"}


def validate(payload: dict) -> dict:
    rows = list(payload.get("results", []))
    ood = [row for row in rows if row.get("metadata", {}).get("stage74_regression_category") == "ood_leak"]
    in_domain = [
        row for row in rows
        if row.get("metadata", {}).get("stage74_expected_decision") == "answered"
    ]
    by_id = {row.get("query_id"): row for row in rows}
    run_success = sum(row.get("status") == "ok" for row in rows) / len(rows) if rows else 0.0
    ood_fallback = sum(bool(row.get("is_fallback")) for row in ood) / len(ood) if ood else 0.0
    in_answer = sum(not row.get("is_fallback") for row in in_domain) / len(in_domain) if in_domain else 0.0
    recovered = sum(not by_id.get(query_id, {}).get("is_fallback", True) for query_id in FALSE_FALLBACK_IDS)
    unknown = sum(len(row.get("unknown_citation_ids", [])) for row in rows)
    metrics = {
        "row_count": len(rows), "run_success_rate": round(run_success, 4),
        "ood_fallback_accuracy": round(ood_fallback, 4),
        "in_domain_answer_rate": round(in_answer, 4),
        "labeled_false_fallback_recovered": recovered,
        "unknown_citation_count": unknown,
    }
    checks = {
        "row_count_is_51": len(rows) == 51,
        "run_success_rate_ge_095": run_success >= 0.95,
        "ood_fallback_accuracy_ge_080": ood_fallback >= 0.80,
        "in_domain_answer_rate_ge_085": in_answer >= 0.85,
        "four_labeled_false_fallbacks_recovered": recovered == 4,
        "unknown_citation_count_is_zero": unknown == 0,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = validate(json.loads(args.input.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
