"""Rule-based query rewriting for Stage5.

The rewriter produces conservative retrieval branches.  It never removes the
raw query and avoids aggressive expansion for exact/keyword queries, so Stage5
can improve scenario / short queries without hurting precise clause lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from rag_v2.query.analyzer import normalize_query_text
from rag_v2.query.models import QueryBranch, QueryPlan

_HAZARD_EXPANSIONS = {
    "台风": ["台风", "热带气旋", "防御措施", "应急处置"],
    "暴雨": ["暴雨", "洪涝", "防汛", "应急处置"],
    "洪涝": ["洪涝", "洪水", "防汛", "转移避险"],
    "地震": ["地震", "震灾", "疏散", "避险"],
    "地质灾害": ["地质灾害", "滑坡", "泥石流", "避险转移"],
    "山洪": ["山洪", "洪涝", "转移避险", "预警"],
    "气象灾害": ["气象灾害", "预警", "应急响应", "防御措施"],
}

_REGION_EXPANSIONS = {
    "深圳": ["深圳", "深圳市"],
    "广东": ["广东", "广东省"],
    "河南": ["河南", "河南省"],
    "湖北": ["湖北", "湖北省"],
    "四川": ["四川", "四川省"],
    "陕西": ["陕西", "陕西省"],
    "北京": ["北京", "北京市"],
    "天津": ["天津", "天津市"],
    "云南": ["云南", "云南省"],
    "吉林": ["吉林", "吉林省"],
    "西藏": ["西藏", "西藏自治区"],
}

_SCENARIO_SUBJECTS = {
    "学校": ["学校", "师生", "停课", "安全管理"],
    "居民": ["居民", "群众", "转移安置"],
    "高层": ["高层建筑", "居民", "疏散", "避险"],
    "高层建筑": ["高层建筑", "居民", "疏散", "避险"],
    "房屋": ["房屋", "建筑", "风险排查"],
    "企业": ["企业", "生产", "应急处置"],
}

_INTENT_EXPANSIONS = {
    "怎么办": ["应急处置", "防御措施", "避险", "转移安置"],
    "怎么做": ["应急处置", "防御措施", "职责", "措施"],
    "如何": ["如何", "措施", "流程", "要求"],
    "疏散": ["疏散", "撤离", "避险", "转移安置"],
    "撤离": ["撤离", "疏散", "避险", "转移安置"],
    "避险": ["避险", "转移", "安置", "预警"],
}

_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class QueryRewriter:
    """Generate weighted query branches from a QueryPlan."""

    max_branches: int | None = None
    max_query_chars: int = 220
    branch_order: tuple[str, ...] = ("raw", "normalized", "expanded")
    exact_query_types: set[str] = field(default_factory=lambda: {"keyword", "exact_fact"})

    def rewrite(self, plan: QueryPlan) -> list[QueryBranch]:
        weights = plan.retrieval_policy.branch_weights
        limit = self.max_branches or plan.retrieval_policy.max_branches
        branches: list[QueryBranch] = []

        self._add_branch(
            branches,
            QueryBranch(
                branch="raw",
                query=plan.original_query.strip(),
                weight=weights.get("raw", 1.0),
                reason="original_query_always_preserved",
            ),
        )

        if not plan.original_query.strip():
            return branches

        if self._allow_normalized(plan):
            self._add_branch(
                branches,
                QueryBranch(
                    branch="normalized",
                    query=plan.normalized_query,
                    weight=weights.get("normalized", 0.9),
                    reason="text_normalization",
                ),
            )

        if self._allow_expanded(plan):
            expanded = build_expanded_query(plan)
            if expanded:
                self._add_branch(
                    branches,
                    QueryBranch(
                        branch="expanded",
                        query=expanded,
                        weight=weights.get("expanded", 0.7),
                        reason=f"{plan.query_type}_rule_expansion",
                    ),
                )

        ordered = sorted(
            branches,
            key=lambda item: (self.branch_order.index(item.branch) if item.branch in self.branch_order else 99),
        )
        return ordered[: max(1, limit)]

    def _allow_normalized(self, plan: QueryPlan) -> bool:
        if not plan.normalized_query:
            return False
        return meaningful_key(plan.normalized_query) != meaningful_key(plan.original_query)

    def _allow_expanded(self, plan: QueryPlan) -> bool:
        if plan.query_type in self.exact_query_types:
            return False
        return plan.retrieval_policy.use_rewrite or plan.retrieval_policy.use_multi_query

    def _add_branch(self, branches: list[QueryBranch], branch: QueryBranch) -> None:
        query = compact_query(branch.query)[: self.max_query_chars]
        if not query:
            return
        normalized = compact_query(query).lower()
        for existing in branches:
            if compact_query(existing.query).lower() == normalized:
                return
        branches.append(QueryBranch(branch=branch.branch, query=query, weight=branch.weight, reason=branch.reason))


def build_expanded_query(plan: QueryPlan) -> str:
    tokens: list[str] = []
    append_unique(tokens, plan.normalized_query)

    if plan.region_hint:
        for item in _REGION_EXPANSIONS.get(plan.region_hint, [plan.region_hint]):
            append_unique(tokens, item)

    if plan.hazard_hint:
        for item in _HAZARD_EXPANSIONS.get(plan.hazard_hint, [plan.hazard_hint]):
            append_unique(tokens, item)

    for subject, expansions in _SCENARIO_SUBJECTS.items():
        if subject in plan.normalized_query:
            for item in expansions:
                append_unique(tokens, item)

    for trigger, expansions in _INTENT_EXPANSIONS.items():
        if trigger in plan.normalized_query:
            for item in expansions:
                append_unique(tokens, item)

    if plan.query_type == "scenario":
        for item in ["应急预案", "响应流程", "责任主体"]:
            append_unique(tokens, item)
    elif plan.query_type == "multi_hop":
        for item in ["综合评估", "部门职责", "处置流程"]:
            append_unique(tokens, item)
    elif plan.query_type == "ambiguous":
        for item in ["预警", "应急处置", "防灾减灾"]:
            append_unique(tokens, item)

    return compact_query(" ".join(tokens))


def append_unique(tokens: list[str], token: str | None) -> None:
    token = compact_query(token or "")
    if not token:
        return
    seen = {item.lower() for item in tokens}
    if token.lower() not in seen:
        tokens.append(token)


def compact_query(query: str) -> str:
    return _SPACE_RE.sub(" ", str(query or "")).strip()


def meaningful_key(query: str) -> str:
    chars: list[str] = []
    for char in str(query or ""):
        category = unicodedata.category(char)
        if category.startswith(("P", "Z")) or char.isspace():
            continue
        chars.append(char)
    return "".join(chars).lower()
