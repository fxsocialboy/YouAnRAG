import json
from pathlib import Path

from rag_v2.evaluation.models import load_evaluation_queries, validate_evaluation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metadata_sets():
    rows = json.loads((PROJECT_ROOT / "artifacts" / "stage1" / "chunk_metadata.json").read_text(encoding="utf-8"))
    return {row["chunk_id"] for row in rows}, {row["source_file"] for row in rows}


def test_final_labeled_dataset_is_frozen_and_labels_exist():
    queries = load_evaluation_queries(PROJECT_ROOT / "experiments" / "eval_queries_final_labeled.jsonl")
    chunks, sources = _metadata_sets()
    report = validate_evaluation_dataset(queries, known_chunk_ids=chunks, known_source_files=sources)

    assert len(queries) == 25
    assert report["valid"] is True
    assert report["group_counts"]["out_of_domain"] == 2
    assert report["group_counts"]["multi_hop"] == 3
    assert sum(report["disaster_type_counts"].values()) == 25
    assert report["disaster_type_counts"]["out_of_domain"] == 2
    assert all(item.expected_fallback is not None for item in queries)
    assert all(item.reference_facts for item in queries if not item.expected_fallback)


def test_final_random_dataset_has_expected_fixed_groups():
    queries = load_evaluation_queries(PROJECT_ROOT / "experiments" / "eval_queries_final_random.jsonl")
    report = validate_evaluation_dataset(queries)

    assert report["valid"] is True
    assert len(queries) == 120
    assert report["group_counts"] == {
        "scenario": 40,
        "keyword": 25,
        "short_ambiguous": 20,
        "multi_hop": 15,
        "out_of_domain": 20,
    }
    assert all(item.metadata["seed"] == 20260811 for item in queries)
    assert sum(report["disaster_type_counts"].values()) == 120
    assert report["disaster_type_counts"]["out_of_domain"] == 20
    assert all(item.expected_fallback is True for item in queries if item.query_type == "out_of_domain")
    assert all(item.expected_fallback is None for item in queries if item.query_type != "out_of_domain")
