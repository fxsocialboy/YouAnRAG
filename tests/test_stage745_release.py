import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_stage74_calibration import validate
from validate_stage74_release import audit


def regression_row(index, category, *, fallback, status="ok", unknown=None):
    query_id = f"q{index}"
    if index < 4:
        query_id = ("final_006", "final_014", "final_019", "final_020")[index]
    decision = "fallback" if category == "ood_leak" else "answered"
    return {
        "query_id": query_id, "status": status, "is_fallback": fallback,
        "unknown_citation_ids": unknown or [],
        "metadata": {"stage74_regression_category": category, "stage74_expected_decision": decision},
    }


def test_calibration_acceptance_passes_target_metrics():
    rows = [regression_row(i, "false_rejection", fallback=False) for i in range(16)]
    rows += [regression_row(100 + i, "positive_control", fallback=False) for i in range(10)]
    rows += [regression_row(200 + i, "domain_boundary", fallback=False) for i in range(5)]
    rows += [regression_row(300 + i, "ood_leak", fallback=i >= 3) for i in range(20)]
    report = validate({"results": rows})
    assert report["passed"] is True
    assert report["metrics"]["ood_fallback_accuracy"] == 0.85


def test_calibration_acceptance_rejects_ood_leakage():
    rows = [regression_row(i, "false_rejection", fallback=False) for i in range(31)]
    rows += [regression_row(100 + i, "ood_leak", fallback=False) for i in range(20)]
    report = validate({"results": rows})
    assert report["passed"] is False
    assert report["checks"]["ood_fallback_accuracy_ge_080"] is False


def test_release_audit_verifies_required_files_and_before_fix():
    report = audit(run_tests=False)
    assert report["passed"] is True
    assert any(item["name"] == "before_fix_integrity" for item in report["checks"])


def test_full_autodl_run_requires_calibration_and_runs_three_langgraph_smokes():
    script = (SCRIPTS / "run_stage7_autodl.sh").read_text(encoding="utf-8")
    assert "calibration_acceptance.json" in script
    assert "--real-llm --limit 3" in script
    assert "--count 15" in script


def test_package_script_uses_stage74_name_and_release_audit():
    script = (SCRIPTS / "package_stage7_autodl.ps1").read_text(encoding="utf-8")
    assert "newrag_stage74_after_fix_code.tar.gz" in script
    assert "validate_stage74_release.py" in script


def test_autodl_package_bundles_minimal_original_langgraph_agent():
    agent = ROOT / "legacy_snapshot/multi_agent_server/app/agents/disaster_response_agent.py"
    smoke = (SCRIPTS / "smoke_youan_agent_stage7.py").read_text(encoding="utf-8")
    assert agent.is_file()
    assert "BUNDLED_SERVER_ROOT" in smoke
    assert 'model_path=embedding_path if embedding_path.exists() else cfg.model_path' in smoke
