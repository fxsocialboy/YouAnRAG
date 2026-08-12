import json
from pathlib import Path

from rag_v2.agent.models import Citation, EvidenceItem
from rag_v2.agent.verifier import CitationVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEFORE_FIX = PROJECT_ROOT / "artifacts" / "stage7" / "before_fix" / "final_labeled_v2_eval.json"


def test_four_labeled_false_fallbacks_are_no_longer_hard_rejected():
    rows = json.loads(BEFORE_FIX.read_text(encoding="utf-8"))["results"]
    by_id = {row["query_id"]: row for row in rows}
    verifier = CitationVerifier()

    for query_id in ("final_006", "final_014", "final_019", "final_020"):
        row = by_id[query_id]
        draft = row["trace"]["extra"]["composer"]["answer"]
        citations = [Citation.from_dict(item) for item in row["citations"]]
        evidence = [EvidenceItem.from_dict(item) for item in row["evidence"]]
        _answer, result = verifier.verify_with_repair(draft, citations, evidence)
        assert result.passed, (query_id, result.to_dict())
        assert "key_fact_without_citation" not in result.reasons


def test_before_fix_mmr_one_does_not_become_retrieval_confidence_one():
    rows = json.loads(BEFORE_FIX.read_text(encoding="utf-8"))["results"]
    evidence = rows[0]["evidence"][0]
    assert evidence["metadata"]["mmr_score"] == 1.0
    # The original reranker value proves that 1.0 was merely the normalized
    # MMR selection score, not an absolute confidence probability.
    assert evidence["metadata"]["rerank_score"] != evidence["metadata"]["mmr_score"]
