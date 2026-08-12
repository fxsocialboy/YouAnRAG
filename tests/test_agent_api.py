def test_agent_api_module_imports_without_loading_models():
    import rag_v2.agent.api as api

    assert hasattr(api, "create_app")
    assert hasattr(api, "AnswerRequest")
