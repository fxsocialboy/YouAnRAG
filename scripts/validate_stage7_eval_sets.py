"""Validate frozen Stage7 datasets before local or AutoDL evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.evaluation.models import load_evaluation_queries, validate_evaluation_dataset


def validate_stage7_datasets(project_root: Path = PROJECT_ROOT) -> dict:
    metadata = json.loads(
        (project_root / "artifacts" / "stage1" / "chunk_metadata.json").read_text(encoding="utf-8")
    )
    known_chunks = {row["chunk_id"] for row in metadata}
    known_sources = {row["source_file"] for row in metadata}
    labeled = load_evaluation_queries(project_root / "experiments" / "eval_queries_final_labeled.jsonl")
    random_set = load_evaluation_queries(project_root / "experiments" / "eval_queries_final_random.jsonl")
    labeled_report = validate_evaluation_dataset(
        labeled, known_chunk_ids=known_chunks, known_source_files=known_sources
    )
    random_report = validate_evaluation_dataset(random_set)
    expected_random_groups = {
        "scenario": 40,
        "keyword": 25,
        "short_ambiguous": 20,
        "multi_hop": 15,
        "out_of_domain": 20,
    }
    checks = {
        "labeled_count_is_25": len(labeled) == 25,
        "labeled_labels_resolve": labeled_report["valid"],
        "random_count_is_120": len(random_set) == 120,
        "random_groups_match": random_report["group_counts"] == expected_random_groups,
        "random_rows_unique": random_report["valid"],
    }
    return {
        "stage": "7.1",
        "labeled": labeled_report,
        "random": random_report,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage7" / "dataset_validation.json",
    )
    args = parser.parse_args()
    report = validate_stage7_datasets()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
