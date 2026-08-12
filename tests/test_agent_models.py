import json

import pytest

from rag_v2.agent.models import AnswerTrace, Citation, EvidenceItem, RagAnswer


def test_evidence_item_normalizes_citation_and_exports_marker():
    evidence = EvidenceItem(
        citation_id="[S1]",
        chunk_id="深圳市气象灾害应急预案.md::48",
        source_file="深圳市气象灾害应急预案.md",
        section_path=["3.3.3.2 Ⅳ级应急响应（一般）"],
        content="由市气象灾害指挥部决定是否启动Ⅳ级应急响应。",
        score="0.87",
        rank=1,
    )

    assert evidence.citation_id == "S1"
    assert evidence.marker == "[S1]"
    assert evidence.section_path_text == "3.3.3.2 Ⅳ级应急响应（一般）"
    assert evidence.score == pytest.approx(0.87)


def test_evidence_item_to_citation_keeps_source_mapping():
    evidence = EvidenceItem(
        citation_id="S2",
        chunk_id="国家气象灾害应急预案-2.md::16",
        source_file="国家气象灾害应急预案-2.md",
        section_path=["（一）IV 级响应"],
        content="签署启动或变更到 IV 级应急响应命令后，局应急办传达命令。",
        score=0.71,
        rank=2,
        metadata={"branch": "raw"},
    )

    citation = evidence.to_citation()

    assert citation.citation_id == "S2"
    assert citation.marker == "[S2]"
    assert citation.chunk_id == evidence.chunk_id
    assert citation.source_file == evidence.source_file
    assert citation.metadata["content_preview"].startswith("签署启动")
    assert citation.metadata["branch"] == "raw"


def test_rag_answer_round_trip_is_json_serializable():
    evidence = EvidenceItem(
        citation_id="S1",
        chunk_id="a.md::0",
        source_file="a.md",
        section_path=["一、总则"],
        content="学校应停止户外活动。",
        score=0.9,
        rank=1,
    )
    answer = RagAnswer(
        query="台风黄色预警下学校应该怎么做",
        answer="学校应停止户外活动。[S1]",
        citations=[evidence.to_citation()],
        evidence=[evidence],
        trace=AnswerTrace(
            query_plan={"query_type": "scenario"},
            branches=[{"branch": "raw"}],
            matched_branches=["raw"],
            retrieval_latency_ms=12.5,
            verification={"passed": True},
        ),
    )

    payload = answer.to_dict()
    dumped = json.dumps(payload, ensure_ascii=False)
    restored = RagAnswer.from_dict(json.loads(dumped))

    assert restored.query == answer.query
    assert restored.answer == answer.answer
    assert restored.citations[0].marker == "[S1]"
    assert restored.evidence[0].section_path_text == "一、总则"
    assert restored.trace.query_plan["query_type"] == "scenario"
    assert restored.is_fallback is False


def test_rag_answer_supports_empty_fallback_response():
    answer = RagAnswer(
        query="火星地震如何处理",
        answer="当前知识库没有足够依据回答该问题。",
        fallback_reason="insufficient_evidence",
    )

    assert answer.is_fallback is True
    assert answer.citations == []
    assert answer.evidence == []
    assert answer.to_dict()["fallback_reason"] == "insufficient_evidence"


def test_invalid_evidence_rejects_empty_required_fields():
    with pytest.raises(ValueError):
        EvidenceItem(
            citation_id="S1",
            chunk_id="",
            source_file="a.md",
            section_path=[],
            content="content",
        )


def test_citation_from_dict_ignores_derived_fields():
    citation = Citation.from_dict(
        {
            "citation_id": "S3",
            "source_file": "b.md",
            "chunk_id": "b.md::1",
            "section_path": ["二、响应"],
            "marker": "[S3]",
            "section_path_text": "二、响应",
        }
    )

    assert citation.marker == "[S3]"
    assert citation.section_path_text == "二、响应"
