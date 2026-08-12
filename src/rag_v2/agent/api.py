"""Optional FastAPI app for Stage6 Agentic RAG demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions


@dataclass(slots=True)
class AnswerRequest:
    query: str
    top_k: int = 5
    hyde_mode: str = "disabled"
    rerank: bool = True
    mmr: bool = True
    composer_mode: Literal["template", "deepseek"] = "template"


def create_app(service: RagAnswerService | None = None):
    """Create a FastAPI app without loading retrieval models at import time."""

    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("FastAPI demo requires fastapi and pydantic to be installed") from exc

    class AnswerRequestModel(BaseModel):
        query: str = Field(..., min_length=1)
        top_k: int = Field(5, ge=1, le=20)
        hyde_mode: str = "disabled"
        rerank: bool = True
        mmr: bool = True
        composer_mode: Literal["template", "deepseek"] = "template"

    app = FastAPI(title="YouAnRAG Stage7 Agentic RAG Demo")
    state = {"service": service}

    def get_service() -> RagAnswerService:
        if state["service"] is None:
            state["service"] = RagAnswerService.from_config()
        return state["service"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stage": "7"}

    @app.post("/answer")
    def answer_endpoint(payload: AnswerRequestModel) -> dict:
        svc = get_service()
        result = svc.answer(
            payload.query,
            options=RagAnswerServiceOptions(
                top_k=payload.top_k,
                enable_reranker=payload.rerank,
                enable_mmr=payload.mmr,
                composer_mode=payload.composer_mode,
            ),
        )
        return result.to_dict()

    return app


try:
    app = create_app(service=None)
except ImportError:  # pragma: no cover - keeps module importable without FastAPI
    app = None
