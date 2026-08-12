"""Lightweight DeepSeek LLM-as-Judge for grounded answer evaluation."""

from __future__ import annotations

import json
import re
from typing import Protocol

from rag_v2.agent.models import EvidenceItem
from rag_v2.evaluation.models import AnswerJudgeResult, AtomicFactJudgment


JUDGE_SYSTEM_PROMPT = """你是严格的 RAG 答案评测员。请仅依据用户问题和证据评估答案，不使用外部知识。
把答案拆成最小可验证的原子事实，并逐条判断：
- supported：证据是否明确支持该事实；
- cited：事实附近是否有引用标记；
- supporting_citation_ids：真正支持该事实的证据编号；
- reason：简短理由。
同时给出 answer_relevancy（0到1），表示答案对用户问题的回应程度。
只输出合法 JSON，不输出 Markdown。"""


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]], **kwargs) -> str: ...


class DeepSeekAnswerJudge:
    def __init__(self, client: ChatClient, *, max_evidence_chars: int = 1600, max_tokens: int = 1800):
        self.client = client
        self.max_evidence_chars = max_evidence_chars
        self.max_tokens = max_tokens

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        evidence: list[EvidenceItem],
        composer_mode: str,
    ) -> AnswerJudgeResult:
        raw = self.client.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(query, answer, evidence)},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        data = parse_json_object(raw)
        facts = []
        known_citation_ids = {item.citation_id for item in evidence}
        for item in data.get("atomic_facts", []):
            raw_ids = item.get("supporting_citation_ids", [])
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            supporting_ids = [str(value).strip("[] ") for value in raw_ids]
            supported = _as_bool(item.get("supported"))
            invalid_ids = [value for value in supporting_ids if value not in known_citation_ids]
            if supported and (not supporting_ids or invalid_ids):
                supported = False
            reason = str(item.get("reason", ""))
            if invalid_ids:
                reason = (reason + f"；Judge引用不存在的证据：{','.join(invalid_ids)}").strip("；")
            facts.append(
                AtomicFactJudgment(
                    fact=str(item["fact"]),
                    supported=supported,
                    cited=_as_bool(item.get("cited")),
                    supporting_citation_ids=supporting_ids,
                    reason=reason,
                )
            )
        if not facts:
            raise ValueError("judge response contains no atomic_facts")
        return AnswerJudgeResult(
            composer_mode=composer_mode,
            atomic_facts=facts,
            answer_relevancy=float(data["answer_relevancy"]),
            reason=str(data.get("reason", "")),
        )

    def _build_prompt(self, query: str, answer: str, evidence: list[EvidenceItem]) -> str:
        blocks = []
        for item in evidence:
            blocks.append(f"{item.marker} {item.source_file}\n{item.content[:self.max_evidence_chars]}")
        schema = {
            "atomic_facts": [
                {
                    "fact": "答案中的一个最小事实",
                    "supported": True,
                    "cited": True,
                    "supporting_citation_ids": ["S1"],
                    "reason": "证据支持或不支持的理由",
                }
            ],
            "answer_relevancy": 0.0,
            "reason": "整体相关性理由",
        }
        return (
            f"【用户问题】\n{query}\n\n【答案】\n{answer}\n\n【证据】\n"
            + "\n\n".join(blocks)
            + "\n\n【输出JSON结构】\n"
            + json.dumps(schema, ensure_ascii=False)
        )


def parse_json_object(text: str) -> dict:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge response must be a JSON object")
    return data


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是", "支持"}
    return bool(value)
