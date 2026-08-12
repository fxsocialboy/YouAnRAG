"""Three-state domain coverage gate for the disaster-response knowledge base."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal, Protocol

from rag_v2.agent.models import EvidenceItem


DomainDecision = Literal["in_domain", "out_of_domain", "uncertain"]

DOMAIN_TERMS = (
    "灾害", "灾情", "防灾", "减灾", "救灾", "救助", "应急", "预警", "应急响应", "预案",
    "救援", "疏散", "撤离", "避险", "安置", "抢险", "风险普查", "指挥体系", "灾后重建",
    "台风", "暴雨", "洪水", "洪涝", "山洪", "地震", "滑坡", "泥石流", "寒潮", "暴雪",
)
DOMAIN_PATTERNS = (
    re.compile(r"(?:Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|I|II|III|IV|V|一|二|三|四|五)级(?:应急)?响应", re.IGNORECASE),
    re.compile(r"第[一二三四五六七八九十百0-9]+条"),
)
OUT_OF_DOMAIN_TERMS = (
    "Transformer", "稀疏注意力", "4K视频", "剪辑电脑", "糖尿病", "胰岛素", "量化交易", "套利",
    "量子计算", "表面码", "法式可颂", "新能源汽车股票", "Java虚拟机", "垃圾回收器", "围棋强化学习",
    "黑洞", "分布式数据库", "计算机博士", "增肌饮食", "摄影", "景深", "NTFS", "现代诗", "CUDA",
    "矩阵乘法", "科幻电影", "电商平台", "优惠券", "蛋白质折叠",
)


class DomainClassifier(Protocol):
    def classify(self, query: str, evidence: list[EvidenceItem]) -> "DomainGateResult": ...


@dataclass(slots=True)
class DomainGateResult:
    decision: DomainDecision
    decision_source: str
    signals: list[str] = field(default_factory=list)
    confidence: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.decision not in {"in_domain", "out_of_domain", "uncertain"}:
            raise ValueError(f"invalid domain decision: {self.decision}")
        if self.confidence is not None:
            self.confidence = float(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DomainGate:
    classifier: DomainClassifier | None = None
    low_confidence_threshold: float = 0.20
    confident_threshold: float = 0.50

    def evaluate(self, query: str, evidence: list[EvidenceItem]) -> DomainGateResult:
        text = str(query or "").strip()
        domain_signals = [term for term in DOMAIN_TERMS if term.lower() in text.lower()]
        domain_signals.extend(match.group(0) for regex in DOMAIN_PATTERNS for match in regex.finditer(text))
        negative_signals = [term for term in OUT_OF_DOMAIN_TERMS if term.lower() in text.lower()]
        confidence = _max_retrieval_confidence(evidence)

        if domain_signals:
            return DomainGateResult(
                "in_domain", "rule", [f"domain_term:{item}" for item in _unique(domain_signals)], confidence,
                "transparent_domain_signal",
            )
        if negative_signals:
            return DomainGateResult(
                "out_of_domain", "rule", [f"out_of_domain_term:{item}" for item in _unique(negative_signals)], confidence,
                "transparent_out_of_domain_signal",
            )

        preliminary = DomainGateResult(
            "uncertain",
            "retrieval",
            ["no_explicit_domain_signal"],
            confidence,
            "high_score_without_domain_signal" if confidence is not None and confidence >= self.confident_threshold else "ambiguous_domain_coverage",
        )
        if self.classifier is None:
            return preliminary
        try:
            classified = self.classifier.classify(text, evidence)
            classified.signals = preliminary.signals + classified.signals
            classified.confidence = confidence
            return classified
        except Exception as exc:
            preliminary.signals.append(f"classifier_failed:{type(exc).__name__}")
            return preliminary


@dataclass(slots=True)
class DeepSeekDomainClassifier:
    client: Any
    max_evidence_items: int = 3
    max_chars: int = 500

    def classify(self, query: str, evidence: list[EvidenceItem]) -> DomainGateResult:
        snippets = "\n".join(
            f"[{item.citation_id}] {item.source_file}: {item.content[:self.max_chars]}"
            for item in evidence[: self.max_evidence_items]
        )
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": "判断问题是否属于自然灾害应急知识库覆盖范围。只输出JSON："
                    '{"decision":"in_domain|out_of_domain|uncertain","reason":"简短原因"}',
                },
                {"role": "user", "content": f"问题：{query}\n检索证据：\n{snippets}"},
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        payload = _parse_json_object(raw)
        decision = str(payload.get("decision", "uncertain"))
        return DomainGateResult(
            decision=decision, decision_source="deepseek", signals=["llm_coverage_check"],
            reason=str(payload.get("reason", "")),
        )


def _max_retrieval_confidence(evidence: list[EvidenceItem]) -> float | None:
    values: list[float] = []
    for item in evidence:
        if item.metadata.get("confidence_type") == "unavailable":
            continue
        value = item.metadata.get("retrieval_confidence", item.score)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("domain classifier response must be a JSON object")
    return payload


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
