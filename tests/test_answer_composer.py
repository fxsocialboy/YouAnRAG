from rag_v2.agent.composer import TemplateAnswerComposer, extract_key_sentence
from rag_v2.agent.models import EvidenceItem


def make_evidence(citation_id="S1", score=0.8, content="学校应停止户外活动。应及时通知家长。"):
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=f"a.md::{citation_id}",
        source_file="深圳市气象灾害应急预案.md",
        section_path=["3.3.3.2 Ⅳ级应急响应（一般）"],
        content=content,
        score=score,
        rank=int(citation_id[1:]),
    )


def test_template_answer_composer_generates_cited_structured_answer():
    composer = TemplateAnswerComposer()
    answer = composer.compose("台风黄色预警下学校应该怎么做", [make_evidence("S1"), make_evidence("S2")])

    assert answer.startswith("根据知识库检索结果")
    assert "处置建议" in answer
    assert "[S1]" in answer
    assert "[S2]" in answer
    assert "来源：" in answer
    assert "深圳市气象灾害应急预案.md" in answer


def test_template_answer_composer_low_confidence_uses_general_advice_header():
    composer = TemplateAnswerComposer(low_confidence_threshold=0.3)
    answer = composer.compose("未知灾害怎么办", [make_evidence("S1", score=0.1)])

    assert answer.startswith("当前知识库未覆盖该问题，以下为一般建议")
    assert "[S1]" in answer


def test_template_answer_composer_empty_evidence_returns_fallback():
    assert TemplateAnswerComposer().compose("火星地震怎么办", []) == "当前知识库没有足够依据回答该问题。"


def test_extract_key_sentence_prefers_policy_sentence_and_truncates():
    content = "背景介绍。" + "学校应立即组织学生转移到安全区域" * 20 + "。"
    sentence = extract_key_sentence(content, max_chars=40)

    assert sentence.startswith("学校应立即组织")
    assert sentence.endswith("……")
    assert len(sentence) <= 41
