from pathlib import Path

from evaluate_stage6 import run_evaluation


def test_stage6_evaluation_closes_agent_contract(tmp_path: Path):
    output = tmp_path / "stage6_eval.json"
    report = run_evaluation(output)

    assert output.exists()
    assert report["summary"]["passed"] is True
    assert report["summary"]["query_count"] == 10
    assert report["summary"]["fallback_count"] == 2
    assert report["summary"]["all_citations_mapped"] is True
    assert report["summary"]["invalid_citation_detected"] is True
    assert report["summary"]["legacy_adapter_ok"] is True
