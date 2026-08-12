"""Build the deterministic Stage 7.4 failure regression set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_v2.evaluation.models import Stage74RegressionQuery, load_evaluation_queries  # noqa: E402


DEFAULT_BASELINE = PROJECT_ROOT / "artifacts" / "stage7" / "before_fix"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "stage74_fix_regression.jsonl"
LABELED_PATH = PROJECT_ROOT / "experiments" / "eval_queries_final_labeled.jsonl"
RANDOM_PATH = PROJECT_ROOT / "experiments" / "eval_queries_final_random.jsonl"

FALSE_REJECTION_QUOTAS = {
    "scenario": 3,
    "keyword": 4,
    "short_ambiguous": 2,
    "multi_hop": 3,
}

DOMAIN_BOUNDARY_ROWS = (
    ("stage74_boundary_001", "气候变化政策中与自然灾害防治直接相关的措施有哪些？", "keyword", "comprehensive"),
    ("stage74_boundary_002", "气象灾害发生后可能有哪些健康影响，应如何防护？", "scenario", "meteorological"),
    ("stage74_boundary_003", "灾害导致应急通信中断时，基层应该如何上报灾情？", "scenario", "comprehensive"),
    ("stage74_boundary_004", "自然灾害灾情统计和损失报告通常要记录哪些信息？", "exact_fact", "comprehensive"),
    ("stage74_boundary_005", "灾后恢复重建阶段应如何安排基础设施修复？", "scenario", "comprehensive"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["results"])


def _row_from_source(
    source: dict[str, Any],
    *,
    source_dataset: str,
    decision: str,
    reason: str,
    category: str,
) -> Stage74RegressionQuery:
    return Stage74RegressionQuery(
        query_id=source["query_id"],
        query=source["query"],
        query_type=source["query_type"],
        disaster_type=source["disaster_type"],
        relevant_source_files=list(source.get("relevant_source_files", [])),
        relevant_chunk_ids=list(source.get("relevant_chunk_ids", [])),
        reference_facts=list(source.get("reference_facts", [])),
        expected_fallback=decision == "fallback",
        metadata={
            **dict(source.get("metadata", {})),
            "stage74_source_kind": "frozen_query",
        },
        expected_decision=decision,
        expected_reason_category=reason,
        regression_category=category,
        source_dataset=source_dataset,
        source_query_id=source["query_id"],
    )


def _select_false_rejections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("query_type") != "out_of_domain" and row.get("is_fallback") is True
    ]
    selected: list[dict[str, Any]] = []
    for query_type, quota in FALSE_REJECTION_QUOTAS.items():
        group = sorted((row for row in candidates if row.get("query_type") == query_type), key=lambda row: row["query_id"])
        if len(group) < quota:
            raise ValueError(f"not enough random false rejections for {query_type}: {len(group)} < {quota}")
        selected.extend(group[:quota])
    return selected


def _select_positive_controls(rows: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("query_type") != "out_of_domain"
        and row.get("status") == "ok"
        and row.get("is_fallback") is False
        and float(row.get("faithfulness") or 0.0) >= 0.9
        and float(row.get("answer_relevancy") or 0.0) >= 0.8
    ]
    selected: list[dict[str, Any]] = []
    query_type_counts: Counter[str] = Counter()
    disaster_counts: Counter[str] = Counter()
    remaining = sorted(candidates, key=lambda row: row["query_id"])
    while remaining and len(selected) < count:
        best = min(
            remaining,
            key=lambda row: (
                query_type_counts[row["query_type"]] + disaster_counts[row["disaster_type"]],
                query_type_counts[row["query_type"]],
                disaster_counts[row["disaster_type"]],
                row["query_id"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        query_type_counts[best["query_type"]] += 1
        disaster_counts[best["disaster_type"]] += 1
    if len(selected) != count:
        raise ValueError(f"only found {len(selected)} positive controls")
    return selected


def build_regression_set(baseline_dir: Path = DEFAULT_BASELINE) -> list[Stage74RegressionQuery]:
    labeled_results = _load_result(baseline_dir / "final_labeled_v2_eval.json")
    random_results = _load_result(baseline_dir / "final_random_v2_eval.json")
    frozen_labeled = {item.query_id: item.to_dict() for item in load_evaluation_queries(LABELED_PATH)}
    frozen_random = {item.query_id: item.to_dict() for item in load_evaluation_queries(RANDOM_PATH)}

    rows: list[Stage74RegressionQuery] = []
    ood_rows = sorted((row for row in random_results if row.get("query_type") == "out_of_domain"), key=lambda row: row["query_id"])
    if len(ood_rows) != 20:
        raise ValueError(f"expected 20 OOD rows, found {len(ood_rows)}")
    rows.extend(
        _row_from_source(frozen_random[row["query_id"]], source_dataset="final_random", decision="fallback", reason="out_of_domain", category="ood_leak")
        for row in ood_rows
    )

    labeled_false = sorted(
        (row for row in labeled_results if row.get("expected_fallback") is False and row.get("is_fallback") is True),
        key=lambda row: row["query_id"],
    )
    if len(labeled_false) != 4:
        raise ValueError(f"expected 4 labeled false rejections, found {len(labeled_false)}")
    rows.extend(
        _row_from_source(frozen_labeled[row["query_id"]], source_dataset="final_labeled", decision="answered", reason="in_domain", category="false_rejection")
        for row in labeled_false
    )

    random_false = _select_false_rejections(random_results)
    rows.extend(
        _row_from_source(frozen_random[row["query_id"]], source_dataset="final_random", decision="answered", reason="in_domain", category="false_rejection")
        for row in random_false
    )

    rows.extend(
        _row_from_source(frozen_random[row["query_id"]], source_dataset="final_random", decision="answered", reason="in_domain", category="positive_control")
        for row in _select_positive_controls(random_results)
    )

    rows.extend(
        Stage74RegressionQuery(
            query_id=query_id,
            query=query,
            query_type=query_type,
            disaster_type=disaster_type,
            expected_fallback=False,
            metadata={"stage74_source_kind": "explicit_domain_boundary"},
            expected_decision="answered",
            expected_reason_category="in_domain",
            regression_category="domain_boundary",
            source_dataset="domain_boundary",
            source_query_id=query_id,
        )
        for query_id, query, query_type, disaster_type in DOMAIN_BOUNDARY_ROWS
    )
    ids = [row.query_id for row in rows]
    if len(rows) != 51 or len(ids) != len(set(ids)):
        raise ValueError(f"invalid regression set size/ids: rows={len(rows)}, unique={len(set(ids))}")
    return rows


def write_regression_set(output_path: Path = DEFAULT_OUTPUT, baseline_dir: Path = DEFAULT_BASELINE) -> dict[str, Any]:
    rows = build_regression_set(baseline_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "stage": "7.4.1",
        "query_count": len(rows),
        "regression_category_counts": dict(sorted(Counter(row.regression_category for row in rows).items())),
        "source_dataset_counts": dict(sorted(Counter(row.source_dataset for row in rows).items())),
        "decision_counts": dict(sorted(Counter(row.expected_decision for row in rows).items())),
        "source_hashes": {
            str(LABELED_PATH.relative_to(PROJECT_ROOT)): _sha256(LABELED_PATH),
            str(RANDOM_PATH.relative_to(PROJECT_ROOT)): _sha256(RANDOM_PATH),
            "artifacts/stage7/before_fix/final_labeled_v2_eval.json": _sha256(baseline_dir / "final_labeled_v2_eval.json"),
            "artifacts/stage7/before_fix/final_random_v2_eval.json": _sha256(baseline_dir / "final_random_v2_eval.json"),
        },
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": _sha256(output_path),
    }
    report_path = baseline_dir / "stage74_regression_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_regression_set(args.output, args.baseline_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
