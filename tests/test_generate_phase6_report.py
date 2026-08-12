from pathlib import Path

from evaluate_stage6 import run_evaluation
from scripts.generate_phase6_report import generate_phase6_report


def test_generate_phase6_report(tmp_path: Path):
    eval_path = tmp_path / "eval.json"
    report_path = tmp_path / "report.md"
    run_evaluation(eval_path)

    result = generate_phase6_report(eval_path, report_path)

    assert result == report_path
    text = report_path.read_text(encoding="utf-8")
    assert "阶段六工程闭环评测：**通过**" in text
    assert "invalid_citation_detected" in text
    assert "Agentic RAG" in text
