"""Rule-based QueryAnalyzer for Stage5.

The first Stage5 increment intentionally avoids external LLM calls.  It uses
transparent rules to classify the query and produce a retrieval policy.  Later
rewrite / HyDE modules consume this QueryPlan without changing Stage4 behavior.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rag_v2.query.models import QueryPlan, QueryType, RetrievalPolicy

_ROMAN_TRANSLATIONS = {
    "ⅰ": "i",
    "ⅱ": "ii",
    "ⅲ": "iii",
    "ⅳ": "iv",
    "ⅴ": "v",
    "Ⅰ": "I",
    "Ⅱ": "II",
    "Ⅲ": "III",
    "Ⅳ": "IV",
    "Ⅴ": "V",
}

_REGION_ALIASES = {
    "深圳市": "深圳",
    "深圳": "深圳",
    "广东省": "广东",
    "广东": "广东",
    "河南省": "河南",
    "河南": "河南",
    "湖北省": "湖北",
    "湖北": "湖北",
    "四川省": "四川",
    "四川": "四川",
    "陕西省": "陕西",
    "陕西": "陕西",
    "北京市": "北京",
    "北京": "北京",
    "天津市": "天津",
    "天津": "天津",
    "西藏自治区": "西藏",
    "西藏": "西藏",
    "吉林省": "吉林",
    "吉林": "吉林",
    "云南省": "云南",
    "云南": "云南",
}

_HAZARD_ALIASES = {
    "气象灾害": "气象灾害",
    "台风": "台风",
    "热带气旋": "台风",
    "暴雨": "暴雨",
    "洪涝": "洪涝",
    "洪水": "洪涝",
    "山洪": "山洪",
    "地震": "地震",
    "震灾": "地震",
    "滑坡": "地质灾害",
    "泥石流": "地质灾害",
    "地质灾害": "地质灾害",
    "大雾": "气象灾害",
    "暴雪": "气象灾害",
    "寒潮": "气象灾害",
}

_DOCUMENT_TYPES = ["预案", "规划", "报告", "公报", "通知", "规范", "办法", "指南", "案例"]
_SCENARIO_PATTERNS = ["怎么办", "怎么做", "如何", "怎么处置", "如何处置", "怎么避险", "如何避险", "撤离", "疏散", "应采取", "措施", "救援"]
_KEYWORD_PATTERNS = ["第", "条", "章", "附件", "附录", "标准", "阈值", "启动条件", "文件", "规定"]
_MULTI_HOP_PATTERNS = ["分别", "对比", "比较", "同时", "以及", "和", "与", "哪些部门", "跨", "综合"]
_RESPONSE_LEVEL_RE = re.compile(r"(?:iv|iii|ii|i|ⅳ|ⅲ|Ⅱ|Ⅳ|Ⅰ|Ⅲ|一|二|三|四|1|2|3|4)\s*级", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*年?")
_ARTICLE_RE = re.compile(r"第\s*[一二三四五六七八九十百千万\d]+\s*[章节条款]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class QueryAnalyzer:
    """Analyze a raw user query into a conservative Stage5 QueryPlan."""

    max_query_chars: int = 300
    branch_weights: dict[str, float] = field(
        default_factory=lambda: {"raw": 1.0, "normalized": 0.9, "expanded": 0.7, "hyde": 0.6}
    )

    def analyze(self, query: str) -> QueryPlan:
        original_query = str(query or "").strip()
        normalized_query = normalize_query_text(original_query)[: self.max_query_chars]
        reasons: list[str] = []

        if not normalized_query:
            policy = RetrievalPolicy(
                use_rewrite=False,
                use_multi_query=False,
                use_hyde=False,
                use_reranker=False,
                use_mmr=False,
                branch_weights=dict(self.branch_weights),
            )
            return QueryPlan(
                original_query=original_query,
                normalized_query="",
                query_type="ambiguous",
                rewrite_confidence=0.0,
                retrieval_policy=policy,
                reasons=["empty_query"],
            )

        region = detect_region_hint(normalized_query)
        hazard = detect_hazard_hint(normalized_query)
        document_type = detect_document_type_hint(normalized_query)
        time_hint = detect_time_hint(normalized_query)
        extracted_terms = extract_terms(normalized_query, region, hazard, document_type, time_hint)
        query_type = classify_query(normalized_query, reasons)
        confidence = estimate_confidence(normalized_query, query_type, extracted_terms)
        policy = build_policy(query_type=query_type, confidence=confidence, branch_weights=self.branch_weights)
        return QueryPlan(
            original_query=original_query,
            normalized_query=normalized_query,
            query_type=query_type,
            region_hint=region,
            hazard_hint=hazard,
            document_type_hint=document_type,
            time_hint=time_hint,
            extracted_terms=extracted_terms,
            rewrite_confidence=confidence,
            retrieval_policy=policy,
            reasons=reasons,
        )


def normalize_query_text(query: str) -> str:
    text = unicodedata.normalize("NFKC", str(query or "")).strip()
    for src, dst in _ROMAN_TRANSLATIONS.items():
        text = text.replace(src, dst)
    text = text.replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV")
    text = text.replace("１", "1").replace("２", "2").replace("３", "3").replace("４", "4")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def detect_region_hint(query: str) -> str | None:
    for alias, canonical in _REGION_ALIASES.items():
        if alias in query:
            return canonical
    return None


def detect_hazard_hint(query: str) -> str | None:
    for alias, canonical in _HAZARD_ALIASES.items():
        if alias in query:
            return canonical
    return None


def detect_document_type_hint(query: str) -> str | None:
    for token in _DOCUMENT_TYPES:
        if token in query:
            return token
    return None


def detect_time_hint(query: str) -> str | None:
    match = _YEAR_RE.search(query)
    return match.group(0).strip() if match else None


def extract_terms(query: str, *hints: str | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint and hint not in seen:
            seen.add(hint)
            terms.append(hint)
    for regex in [_RESPONSE_LEVEL_RE, _ARTICLE_RE, _YEAR_RE]:
        for match in regex.finditer(query):
            token = match.group(0).strip()
            if token and token not in seen:
                seen.add(token)
                terms.append(token)
    for token in ["应急响应", "启动", "疏散", "避险", "损失", "学校", "居民", "高层建筑"]:
        if token in query and token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def classify_query(query: str, reasons: list[str] | None = None) -> QueryType:
    reasons = reasons if reasons is not None else []
    lower = query.lower()
    has_response_level = bool(_RESPONSE_LEVEL_RE.search(lower))
    has_year = bool(_YEAR_RE.search(query))
    has_article = bool(_ARTICLE_RE.search(query))
    has_number = bool(_NUMBER_RE.search(query))
    has_scenario = any(token in query for token in _SCENARIO_PATTERNS)
    has_keyword = any(token in query for token in _KEYWORD_PATTERNS)
    multi_hits = sum(1 for token in _MULTI_HOP_PATTERNS if token in query)

    if has_article or has_keyword and (has_response_level or has_year or has_number):
        reasons.append("keyword_or_article_constraint")
        return "keyword"
    if has_response_level or has_year or (has_number and any(token in query for token in ["损失", "震级", "级以上", "人数", "比例"])):
        reasons.append("exact_fact_constraint")
        return "exact_fact"
    if has_scenario:
        if multi_hits >= 2:
            reasons.append("scenario_multi_constraint")
            return "multi_hop"
        reasons.append("scenario_intent")
        return "scenario"
    if multi_hits >= 2:
        reasons.append("multi_hop_terms")
        return "multi_hop"
    if len(query) <= 8:
        reasons.append("short_ambiguous_query")
        return "ambiguous"
    if has_keyword:
        reasons.append("keyword_term")
        return "keyword"
    reasons.append("default_scenario_like")
    return "scenario"


def estimate_confidence(query: str, query_type: QueryType, terms: list[str]) -> float:
    if not query:
        return 0.0
    base = {
        "keyword": 0.86,
        "exact_fact": 0.88,
        "scenario": 0.78,
        "multi_hop": 0.72,
        "ambiguous": 0.45,
    }[query_type]
    if terms:
        base += min(0.08, 0.02 * len(terms))
    if len(query) <= 6:
        base -= 0.1
    return round(max(0.0, min(0.98, base)), 2)


def build_policy(*, query_type: QueryType, confidence: float, branch_weights: dict[str, float]) -> RetrievalPolicy:
    common = dict(branch_weights=dict(branch_weights))
    if query_type in {"keyword", "exact_fact"}:
        return RetrievalPolicy(
            use_rewrite=False,
            use_multi_query=False,
            use_hyde=False,
            use_reranker=False,
            use_mmr=False,
            dense_top_k=30,
            sparse_top_k=40,
            rerank_top_k=20,
            max_branches=2,
            **common,
        )
    if query_type == "scenario":
        return RetrievalPolicy(
            use_rewrite=True,
            use_multi_query=True,
            use_hyde=False,
            use_reranker=True,
            use_mmr=True,
            dense_top_k=35,
            sparse_top_k=30,
            rerank_top_k=30,
            max_branches=3,
            **common,
        )
    if query_type == "multi_hop":
        return RetrievalPolicy(
            use_rewrite=True,
            use_multi_query=True,
            use_hyde=False,
            use_reranker=True,
            use_mmr=True,
            dense_top_k=40,
            sparse_top_k=35,
            rerank_top_k=35,
            max_branches=3,
            **common,
        )
    return RetrievalPolicy(
        use_rewrite=confidence >= 0.4,
        use_multi_query=confidence >= 0.4,
        use_hyde=False,
        use_reranker=True,
        use_mmr=False,
        dense_top_k=30,
        sparse_top_k=30,
        rerank_top_k=25,
        max_branches=2,
        **common,
    )
