import sys
from pathlib import Path

from rag_v2.agent.legacy_adapter import RagV2RetrieverAdapter
from rag_v2.agent.models import AnswerTrace, Citation, EvidenceItem, RagAnswer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SERVER_ROOT = PROJECT_ROOT.parent / "Youan-AI-main" / "youan-multiagent" / "multi_agent_server"
BUNDLED_SERVER_ROOT = PROJECT_ROOT / "legacy_snapshot" / "multi_agent_server"
OLD_SERVER_ROOT = EXTERNAL_SERVER_ROOT if EXTERNAL_SERVER_ROOT.exists() else BUNDLED_SERVER_ROOT
if str(OLD_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(OLD_SERVER_ROOT))

from app.agents.disaster_response_agent import AgentNodes, DisasterResponseAgent


class FakeAnswerService:
    def answer(self, query, *, options=None):
        evidence = EvidenceItem(
            "S1",
            "policy.md::1",
            "policy.md",
            ["学校响应"],
            "台风黄色预警时，学校应停止户外活动。",
            0.9,
            1,
        )
        return RagAnswer(
            query=query,
            answer="学校应停止户外活动。[S1]",
            citations=[evidence.to_citation()],
            evidence=[evidence],
            trace=AnswerTrace(query_plan={"query_type": "scenario"}, branches=[{"branch": "raw"}]),
        )


class FakeLegacyRetriever:
    backend_name = "legacy"

    def invoke(self, query):
        return ["legacy chunk"]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke_messages(self, messages, **kwargs):
        self.calls.append(messages)
        system = messages[0]["content"]
        if "评审委员会" in system:
            return "方案整体质量很高，没有明显的修改建议。"
        assert "[S1]" in messages[1]["content"]
        return "学校应停止户外活动，并组织学生避险。[S1]"


def test_v2_retrieve_node_puts_citations_and_trace_into_graph_state():
    adapter = RagV2RetrieverAdapter(FakeAnswerService())
    nodes = AgentNodes(adapter, FakeLLM(), FakeLLM())

    output = nodes.retrieve_node({"prompt": "台风黄色预警下学校怎么办"})

    assert output["rag_backend"] == "v2"
    assert output["documents"][0].startswith("[S1]\n来源：policy.md")
    assert output["citations"][0]["chunk_id"] == "policy.md::1"
    assert output["retrieval_trace"]["query_plan"]["query_type"] == "scenario"


def test_legacy_retrieve_node_contract_still_works():
    nodes = AgentNodes(FakeLegacyRetriever(), FakeLLM(), FakeLLM())

    output = nodes.retrieve_node({"prompt": "query"})

    assert output["documents"] == ["legacy chunk"]
    assert output["rag_backend"] == "legacy"
    assert output["citations"] == []


def test_complete_langgraph_runs_with_v2_state_and_fake_llms():
    proposer, critic = FakeLLM(), FakeLLM()
    agent = DisasterResponseAgent(
        retriever=RagV2RetrieverAdapter(FakeAnswerService()),
        proposer_llm=proposer,
        critique_llm=critic,
        max_iterations=1,
    )

    result = agent.generate_plan_with_trace("台风黄色预警下学校怎么办", verbose=False)

    assert result["rag_backend"] == "v2"
    assert "[S1]" in result["plan"]
    assert result["citations"][0]["source_file"] == "policy.md"
    assert result["retrieval_trace"]["branches"][0]["branch"] == "raw"
    assert result["iterations_count"] == 1
    assert len(proposer.calls) == 2  # propose + finalize
    assert len(critic.calls) == 1


def test_pyproject_declares_installable_src_package():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "youan-rag-v2"' in text
    assert 'package-dir = {"" = "src"}' in text
