import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_legacy_hash import _resolve_legacy_path, check
from preflight_stage74_autodl import run as preflight


def test_legacy_hash_paths_are_forced_under_project_snapshot():
    resolved = _resolve_legacy_path(r"G:\old\newrag\legacy_snapshot\RAG\search.py")
    assert resolved == PROJECT_ROOT / "legacy_snapshot" / "RAG" / "search.py"
    assert _resolve_legacy_path(r"G:\outside\secret.py") is None


def test_legacy_hash_failure_is_report_data_not_process_failure(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"path": "legacy_snapshot/RAG/missing.py", "sha256": "0"}]), encoding="utf-8")
    report = check(manifest)
    assert report["passed"] is False
    assert report["items"][0]["status"] == "missing"


def test_preflight_can_run_locally_without_requiring_cuda():
    report = preflight(require_cuda=False, check_deepseek=False)
    names = {item["name"] for item in report["checks"]}
    assert "cuda" in names
    assert "artifacts/stage3/bm25_index.json" in names
    assert "qdrant_local" in names


def test_autodl_script_is_after_fix_isolated_and_packages_manifest():
    script = (SCRIPTS / "run_stage7_autodl.sh").read_text(encoding="utf-8")
    assert 'OUT="artifacts/stage7/after_fix"' in script
    assert "package_manifest.json" in script
    assert "stage74_after_fix_results.tar.gz" in script
    assert "[OK] Stage7.4 after-fix evaluation finished" in script
    assert script.index("import_stage1_to_qdrant.py") < script.index("preflight_stage74_autodl.py")
