from collections import Counter
import json
from pathlib import Path

from rag_v2.evaluation.models import load_evaluation_queries, load_stage74_regression_queries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_PATH = PROJECT_ROOT / "experiments" / "stage74_fix_regression.jsonl"


def test_stage74_regression_schema_and_category_counts():
    rows = load_stage74_regression_queries(REGRESSION_PATH)
    assert len(rows) == 51
    assert len({row.query_id for row in rows}) == 51
    assert Counter(row.regression_category for row in rows) == {
        "ood_leak": 20,
        "false_rejection": 16,
        "positive_control": 10,
        "domain_boundary": 5,
    }
    # The adapter makes the same rows consumable by fake and real evaluators.
    assert all(row.to_evaluation_query().query for row in rows)


def test_stage74_rows_have_frozen_or_explicit_boundary_provenance():
    rows = load_stage74_regression_queries(REGRESSION_PATH)
    frozen = {}
    for filename in ("eval_queries_final_labeled.jsonl", "eval_queries_final_random.jsonl"):
        for item in load_evaluation_queries(PROJECT_ROOT / "experiments" / filename):
            frozen[item.query_id] = item.query
    for row in rows:
        if row.source_dataset == "domain_boundary":
            assert row.query_id.startswith("stage74_boundary_")
            assert row.metadata["stage74_source_kind"] == "explicit_domain_boundary"
        else:
            assert row.source_query_id in frozen
            assert row.query == frozen[row.source_query_id]
            assert row.metadata["stage74_source_kind"] == "frozen_query"


def test_stage74_regression_manifest_matches_file():
    manifest = json.loads(
        (PROJECT_ROOT / "artifacts" / "stage7" / "before_fix" / "stage74_regression_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["query_count"] == 51
    assert manifest["decision_counts"] == {"answered": 31, "fallback": 20}
