"""Template/extractive answer composer for Stage6.

The first Stage6 implementation intentionally avoids LLM generation.  It only
uses text that already appears in evidence chunks, then appends citation markers
so later verifier/guardrail stages can deterministically check the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from rag_v2.agent.models import EvidenceItem


@dataclass(slots=True)
class TemplateAnswerComposer:
    max_evidence_items: int = 4
    low_confidence_threshold: float = 0.3
    fallback_answer: str = "当前知识库没有足够依据回答该问题。"

    def compose(self, query: str, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return self.fallback_answer

        selected = evidence[: self.max_evidence_items]
        low_confidence = all(item.score < self.low_confidence_threshold for item in selected)
        header = "根据知识库检索结果，可以这样处理："
        if low_confidence:
            header = "当前知识库未覆盖该问题，以下为一般建议："

        lines = [header, ""]
        labels = self._labels_for_query(query, len(selected))
        for idx, item in enumerate(selected):
            sentence = extract_key_sentence(item.content)
            lines.append(f"{idx + 1}. {labels[idx]}：{sentence}{item.marker}")

        lines.extend(["", "来源："])
        for item in selected:
            section = f" / {item.section_path_text}" if item.section_path_text else ""
            lines.append(f"{item.marker} {item.source_file}{section}")
        return "\n".join(lines)

    def _labels_for_query(self, query: str, count: int) -> list[str]:
        base = ["处置建议", "启动条件或责任主体", "注意事项", "补充依据"]
        if any(term in query for term in ("谁", "哪个部门", "哪级", "启动")):
            base = ["启动条件或责任主体", "处置建议", "注意事项", "补充依据"]
        if any(term in query for term in ("来源", "依据", "文件", "规定")):
            base = ["政策依据", "适用范围", "关键条款", "补充依据"]
        while len(base) < count:
            base.append("补充依据")
        return base[:count]


def extract_key_sentence(content: str, *, max_chars: int = 180) -> str:
    """Extract one compact sentence without inventing or rewriting facts."""

    text = normalize_space(content)
    if not text:
        return "知识库片段为空。"
    sentences = [item.strip() for item in re.split(r"(?<=[。！？；;])\s*|\n+", text) if item.strip()]
    sentence = max(sentences, key=_sentence_priority) if sentences else text
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max(0, max_chars - 1)].rstrip("，,；;。 ") + "……"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _sentence_priority(sentence: str) -> tuple[int, int]:
    policy_terms = ("应", "必须", "不得", "启动", "响应", "预警", "组织", "决定", "负责", "报告")
    score = sum(1 for term in policy_terms if term in sentence)
    return score, min(len(sentence), 220)
