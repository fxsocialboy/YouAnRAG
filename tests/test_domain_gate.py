import json
from pathlib import Path

from rag_v2.agent.domain_gate import DeepSeekDomainClassifier, DomainGate, DomainGateResult
from rag_v2.agent.models import EvidenceItem
from rag_v2.evaluation.models import load_stage74_regression_queries


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evidence(score=0.9):
    return [
        EvidenceItem(
            citation_id="S1",
            chunk_id="doc.md::1",
            source_file="doc.md",
            section_path=[],
            content="应急响应启动后应及时上报灾情。",
            score=score,
            rank=1,
            metadata={"retrieval_confidence": score, "confidence_type": "rerank"},
        )
    ]


class FakeClassifier:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def classify(self, query, items):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


def test_domain_gate_recognizes_response_question_without_hazard_name():
    result = DomainGate().evaluate("四级响应到底由谁启动？", evidence())
    assert result.decision == "in_domain"
    assert result.decision_source == "rule"


def test_domain_gate_does_not_trust_high_rerank_lexical_collision():
    classifier = FakeClassifier(DomainGateResult("out_of_domain", "deepseek", ["semantic_mismatch"], reason="not covered"))
    result = DomainGate(classifier=classifier).evaluate("如何设计一个低延迟的数据库？", evidence(0.99))
    assert result.decision == "out_of_domain"
    assert result.confidence == 0.99
    assert classifier.calls == 1


def test_domain_gate_degrades_to_uncertain_when_classifier_fails():
    result = DomainGate(classifier=FakeClassifier(error=TimeoutError())).evaluate("这个问题怎么处理？", evidence())
    assert result.decision == "uncertain"
    assert "classifier_failed:TimeoutError" in result.signals


def test_deepseek_domain_classifier_uses_json_protocol():
    client = FakeClient(json.dumps({"decision": "out_of_domain", "reason": "非自然灾害问题"}))
    result = DeepSeekDomainClassifier(client).classify("写排序算法", evidence())
    assert result.decision == "out_of_domain"
    assert client.calls[0][1]["response_format"] == {"type": "json_object"}


def test_all_twenty_ood_regression_queries_are_rejected_by_transparent_rules():
    rows = load_stage74_regression_queries(PROJECT_ROOT / "experiments" / "stage74_fix_regression.jsonl")
    ood = [row for row in rows if row.regression_category == "ood_leak"]
    decisions = [DomainGate().evaluate(row.query, evidence()).decision for row in ood]
    assert len(ood) == 20
    assert decisions.count("out_of_domain") / len(decisions) >= 0.80
