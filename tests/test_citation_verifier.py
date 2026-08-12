from rag_v2.agent.models import Citation, EvidenceItem
from rag_v2.agent.verifier import CitationVerifier


def make_evidence(citation_id="S1", score=0.8):
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=f"a.md::{citation_id}",
        source_file="a.md",
        section_path=["一、响应"],
        content="由市气象灾害指挥部决定是否启动Ⅳ级应急响应。",
        score=score,
        rank=int(citation_id[1:]),
    )


def test_citation_verifier_accepts_valid_answer():
    evidence = [make_evidence("S1")]
    citations = [item.to_citation() for item in evidence]

    result = CitationVerifier().verify("启动Ⅳ级应急响应应由相关指挥机构决定。[S1]", citations, evidence)

    assert result.passed is True
    assert result.used_citations == ["S1"]
    assert result.reasons == []


def test_citation_verifier_detects_missing_citation_id():
    result = CitationVerifier().verify("启动Ⅳ级响应。[S9]", [], [])

    assert result.passed is False
    assert "missing_citation" in result.reasons
    assert result.missing_citations == ["S9"]


def test_citation_verifier_detects_key_fact_without_citation():
    evidence = [make_evidence("S1")]
    citations = [item.to_citation() for item in evidence]

    result = CitationVerifier().verify("启动Ⅳ级响应应由相关指挥机构决定。", citations, evidence)

    assert result.passed is True
    assert "key_fact_without_citation" not in result.reasons
    assert "key_fact_without_citation" in result.warnings
    assert result.has_key_fact_without_citation is True


def test_citation_verifier_detects_citation_without_evidence():
    citation = Citation(citation_id="S1", source_file="a.md", chunk_id="a.md::1")

    result = CitationVerifier().verify("处置建议。[S1]", [citation], [])

    assert result.passed is False
    assert "citation_without_evidence" in result.reasons


def test_citation_verifier_ignores_generic_number_heading_and_list_number():
    result = CitationVerifier().verify(
        "# 4K视频处置说明\n1.\n级别不同，措施也不同。",
        [],
        [],
    )

    assert result.passed is True
    assert result.has_key_fact_without_citation is False


def test_citation_verifier_keeps_unknown_reference_as_hard_failure():
    evidence = [make_evidence("S1")]
    result = CitationVerifier().verify("启动Ⅳ级响应。[S99]", [evidence[0].to_citation()], evidence)

    assert result.passed is False
    assert result.missing_citations == ["S99"]


def test_citation_verifier_can_repair_one_exact_fact_with_existing_marker():
    evidence = [make_evidence("S1")]
    citations = [evidence[0].to_citation()]

    answer, result = CitationVerifier().verify_with_repair(
        "应由指挥部决定是否启动Ⅳ级应急响应。",
        citations,
        evidence,
    )

    assert answer.endswith("[S1]")
    assert result.passed is True
    assert result.warnings == []


def test_evidence_without_citation_is_diagnostic_not_hard_failure():
    evidence = [make_evidence("S1")]
    result = CitationVerifier().verify("没有引用的概括性回答。", [], evidence)

    assert result.passed is True
    assert result.diagnostics == ["evidence_without_citation"]
