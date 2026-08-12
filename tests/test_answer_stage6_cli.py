import importlib.util
from pathlib import Path


def load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "answer_stage6.py"
    spec = importlib.util.spec_from_file_location("answer_stage6", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_answer_stage6_parse_args_and_build_options():
    module = load_cli_module()
    args = module.parse_args(["台风怎么办", "--top-k", "3", "--no-rerank", "--no-mmr", "--min-evidence-score", "0.4"])
    options = module.build_options(args)

    assert args.query == "台风怎么办"
    assert options.top_k == 3
    assert options.enable_reranker is False
    assert options.enable_mmr is False
    assert options.min_evidence_score == 0.4
