"""Legacy-compatible adapter for future integration with the original R Agent.

The original RAG component exposes an invoke(query, top_k) -> list[str]-like
interface.  Stage6 keeps that compatibility while also exposing the richer
RagAnswer object for new Agent nodes.
"""

from __future__ import annotations

from typing import Protocol

from rag_v2.agent.models import RagAnswer
from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions


class AnswerService(Protocol):
    def answer(self, query: str, *, options: RagAnswerServiceOptions | None = None) -> RagAnswer: ...


_default_service: AnswerService | None = None


class RagV2RetrieverAdapter:
    """Injectable V2 retriever for the original DisasterResponseAgent."""

    backend_name = "v2"

    def __init__(self, service: AnswerService, *, top_k: int = 10):
        self.service = service
        self.top_k = top_k

    @classmethod
    def from_config(
        cls,
        *,
        top_k: int = 10,
        device: str = "cpu",
        batch_size: int = 16,
        reranker_device: str | None = None,
        hyde_mode: str = "disabled",
        enable_reranker: bool = True,
        enable_mmr: bool = True,
        cfg=None,
    ) -> "RagV2RetrieverAdapter":
        service = RagAnswerService.from_config(
            cfg=cfg,
            device=device,
            batch_size=batch_size,
            reranker_device=reranker_device,
            hyde_mode=hyde_mode,
            enable_reranker=enable_reranker,
            enable_mmr=enable_mmr,
        )
        return cls(service, top_k=top_k)

    def answer(self, query: str, top_k: int | None = None) -> RagAnswer:
        return self.service.answer(
            query,
            options=RagAnswerServiceOptions(top_k=top_k or self.top_k, composer_mode="template"),
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RagAnswer:
        return self.answer(query, top_k=top_k)

    def invoke(self, query: str, top_k: int | None = None) -> list[str]:
        result = self.answer(query, top_k=top_k)
        limit = top_k or self.top_k
        return [item.content for item in result.evidence[:limit]]


def set_default_service(service: AnswerService | None) -> None:
    """Set or reset the process-local default service.

    Tests and demos can inject a fake service.  Production code can leave it as
    None, in which case the real Stage5 stack is lazily built on first use.
    """

    global _default_service
    _default_service = service


def get_default_service() -> AnswerService:
    global _default_service
    if _default_service is None:
        _default_service = RagAnswerService.from_config()
    return _default_service


def answer(query: str, top_k: int = 10, *, service: AnswerService | None = None) -> RagAnswer:
    """Return the full Stage6 answer object with citations/evidence/trace."""

    active_service = service or get_default_service()
    return active_service.answer(query, options=RagAnswerServiceOptions(top_k=top_k))


def invoke(query: str, top_k: int = 10, *, service: AnswerService | None = None) -> list[str]:
    """Compatibility wrapper matching the original RAG.invoke style.

    It returns raw evidence chunk texts, not the generated answer.  This allows
    the old retrieve_node to keep receiving a list[str] before GraphState is
    expanded in a later stage.
    """

    result = answer(query, top_k=top_k, service=service)
    return [item.content for item in result.evidence[:top_k]]
