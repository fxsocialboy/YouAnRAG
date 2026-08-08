from pathlib import Path

from scripts.generate_phase1_report import generate_phase1_report


def test_generate_phase1_report_writes_closing_docs():
    phase1_report, artifact_readme = generate_phase1_report(Path(r"G:\tiaozhanbei\newrag"))
    assert phase1_report.exists()
    assert artifact_readme.exists()
    report_text = phase1_report.read_text(encoding="utf-8")
    readme_text = artifact_readme.read_text(encoding="utf-8")
    assert "阶段一收口报告" in report_text
    assert "Stage1 FAISS 构建摘要" in report_text
    assert "Legacy vs Stage1" in report_text
    assert "6269" in report_text
    assert "Stage1 Artifacts" in readme_text
    assert "faiss_index.index" in readme_text
    assert "GPU" in readme_text
