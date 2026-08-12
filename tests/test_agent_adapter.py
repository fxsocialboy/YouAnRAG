from rag_v2.agent.legacy_adapter import RagV2RetrieverAdapter, answer, invoke, set_default_service
from rag_v2.agent.models import RagAnswer


class FakeService:
    def __init__(self):
        self.calls = []

    def answer(self, query, *, options=None):
        self.calls.append({"query": query, "options": options})
        return RagAnswer(
            query=query,
            answer="结构化答案 [S1]",
            evidence=[
                {
                    "citation_id": "S1",
                    "chunk_id": "a.md::0",
                    "source_file": "a.md",
                    "section_path": [],
                    "content": "chunk text 1",
                    "score": 0.9,
                    "rank": 1,
                },
                {
                    "citation_id": "S2",
                    "chunk_id": "b.md::1",
                    "source_file": "b.md",
                    "section_path": [],
                    "content": "chunk text 2",
                    "score": 0.8,
                    "rank": 2,
                },
            ],
        )


def test_legacy_adapter_answer_returns_full_rag_answer():
    service = FakeService()

    result = answer("台风怎么办", top_k=3, service=service)

    assert isinstance(result, RagAnswer)
    assert result.answer == "结构化答案 [S1]"
    assert service.calls[0]["options"].top_k == 3


def test_legacy_adapter_invoke_returns_evidence_text_list():
    service = FakeService()

    chunks = invoke("台风怎么办", top_k=1, service=service)

    assert chunks == ["chunk text 1"]


def test_legacy_adapter_default_service_can_be_injected_and_reset():
    service = FakeService()
    set_default_service(service)
    try:
        assert invoke("测试", top_k=2) == ["chunk text 1", "chunk text 2"]
    finally:
        set_default_service(None)


def test_object_adapter_supports_legacy_and_structured_interfaces():
    service = FakeService()
    adapter = RagV2RetrieverAdapter(service, top_k=1)

    assert adapter.backend_name == "v2"
    assert adapter.invoke("query") == ["chunk text 1"]
    assert adapter.retrieve("query").evidence[0].citation_id == "S1"
    assert service.calls[-1]["options"].composer_mode == "template"
