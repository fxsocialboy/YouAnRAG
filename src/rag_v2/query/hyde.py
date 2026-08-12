"""HyDE generators for Stage5.

HyDE text is used only as an additional retrieval branch.  It must never be
stored as evidence or citation content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Protocol
from urllib import error, request

from rag_v2.query.models import QueryPlan

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_HYDE_PROMPT = """你是一个应急管理政策专家。请根据以下用户问题，生成一段简短的检索用假设文档（80~150 字），帮助向量检索引擎找到相关政策段落。只输出文档内容，不要任何前缀、后缀或解释。\n\n用户问题：{query}"""


@dataclass(slots=True, frozen=True)
class HydeDocument:
    query: str
    content: str
    used: bool
    reason: str
    latency_ms: float = 0.0
    mode: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HydeGenerator(Protocol):
    mode: str

    def generate(self, query: str, plan: QueryPlan) -> HydeDocument: ...


@dataclass(slots=True)
class DisabledHydeGenerator:
    mode: str = "disabled"

    def generate(self, query: str, plan: QueryPlan) -> HydeDocument:
        return HydeDocument(query=query, content="", used=False, reason="hyde_disabled", mode=self.mode)


@dataclass(slots=True)
class FakeHydeGenerator:
    content: str = "假设文档：用户正在询问灾害场景下的应急处置、预警响应、避险转移和责任主体。"
    mode: str = "fake"

    def generate(self, query: str, plan: QueryPlan) -> HydeDocument:
        if not should_use_hyde(plan):
            return HydeDocument(query=query, content="", used=False, reason=f"query_type_{plan.query_type}_not_allowed", mode=self.mode)
        return HydeDocument(query=query, content=self.content, used=True, reason="fake_hyde", mode=self.mode)


@dataclass(slots=True)
class RuleBasedHydeGenerator:
    mode: str = "rule"

    def generate(self, query: str, plan: QueryPlan) -> HydeDocument:
        started = time.perf_counter()
        if not should_use_hyde(plan):
            return HydeDocument(
                query=query,
                content="",
                used=False,
                reason=f"query_type_{plan.query_type}_not_allowed",
                latency_ms=_elapsed_ms(started),
                mode=self.mode,
            )
        content = build_rule_based_hyde(query, plan)
        return HydeDocument(query=query, content=content, used=bool(content), reason="rule_based_hyde", latency_ms=_elapsed_ms(started), mode=self.mode)


@dataclass(slots=True)
class DeepSeekHydeGenerator:
    api_key: str | None
    model: str = "deepseek-chat"
    timeout: int = 8
    max_retries: int = 1
    fallback: HydeGenerator | None = None
    mode: str = "deepseek"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when hyde_mode=deepseek")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def generate(self, query: str, plan: QueryPlan) -> HydeDocument:
        started = time.perf_counter()
        if not should_use_hyde(plan):
            return HydeDocument(
                query=query,
                content="",
                used=False,
                reason=f"query_type_{plan.query_type}_not_allowed",
                latency_ms=_elapsed_ms(started),
                mode=self.mode,
            )
        try:
            content = self._call_deepseek(query)
            return HydeDocument(query=query, content=content, used=bool(content), reason="deepseek_hyde", latency_ms=_elapsed_ms(started), mode=self.mode)
        except Exception as exc:
            if self.fallback is not None:
                fallback_doc = self.fallback.generate(query, plan)
                return HydeDocument(
                    query=query,
                    content=fallback_doc.content,
                    used=fallback_doc.used,
                    reason=f"deepseek_failed_fallback_{fallback_doc.reason}: {exc.__class__.__name__}",
                    latency_ms=_elapsed_ms(started),
                    mode=f"deepseek->{fallback_doc.mode}",
                )
            return HydeDocument(query=query, content="", used=False, reason=f"deepseek_failed: {exc.__class__.__name__}", latency_ms=_elapsed_ms(started), mode=self.mode)

    def _call_deepseek(self, query: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": DEEPSEEK_HYDE_PROMPT.format(query=query)}],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                req = request.Request(DEEPSEEK_CHAT_COMPLETIONS_URL, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return sanitize_hyde_content(str(content))
            except (error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError("DeepSeek HyDE request failed") from last_error


def should_use_hyde(plan: QueryPlan) -> bool:
    return plan.query_type in {"scenario", "ambiguous"} and not _has_exact_guardrail(plan)


def build_rule_based_hyde(query: str, plan: QueryPlan) -> str:
    hazard = plan.hazard_hint or "灾害"
    region = f"{plan.region_hint}地区" if plan.region_hint else "相关地区"
    terms = "、".join(plan.extracted_terms[:4]) if plan.extracted_terms else "预警、响应、处置、避险"
    if plan.query_type == "ambiguous":
        text = f"{region}发生{hazard}风险时，应查询应急预案中关于{terms}的规定，重点关注预警发布、应急响应启动、人员疏散转移、临时安置和部门职责等政策段落。"
    else:
        text = f"针对用户描述的{hazard}处置场景，应检索应急预案和防灾减灾文件中关于{terms}的内容，包括风险研判、预警响应、学校或居民避险、转移安置、救援协同和责任主体。"
    return sanitize_hyde_content(text)


def sanitize_hyde_content(content: str, *, max_chars: int = 220) -> str:
    cleaned = " ".join(str(content or "").strip().split())
    return cleaned[:max_chars]


def _has_exact_guardrail(plan: QueryPlan) -> bool:
    if plan.query_type in {"keyword", "exact_fact"}:
        return True
    exact_markers = ["第", "条", "章", "附件", "附录"]
    return bool(plan.time_hint) or any(marker in plan.normalized_query for marker in exact_markers)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
