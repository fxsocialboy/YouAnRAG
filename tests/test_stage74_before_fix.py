import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from archive_stage74_before_fix import RESULT_SPECS, verify_before_fix


BASELINE = PROJECT_ROOT / "artifacts" / "stage7" / "before_fix"


def test_before_fix_baseline_has_frozen_counts_and_hashes():
    manifest = json.loads((BASELINE / "before_fix_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "final_labeled_legacy_eval.json": "f492730c1916a1f103ada6ae27882aeb9a8b5e743f4b9484bd38afaa848753c8",
        "final_labeled_v2_eval.json": "18beda840598c931fa95ea783e711bb6635ebc78742d941d6d53950d4c785702",
        "final_random_v2_eval.json": "b20a83479f870d20e248fafa5dda8eb50caa61ced12d0d58cac09662d88c1a30",
    }
    assert set(manifest["results"]) == set(RESULT_SPECS)
    for name, count in RESULT_SPECS.items():
        assert manifest["results"][name]["query_count"] == count
        assert manifest["archived_files"][name]["sha256"] == expected_hashes[name]
        assert hashlib.sha256((BASELINE / name).read_bytes()).hexdigest() == expected_hashes[name]


def test_before_fix_can_be_verified_with_one_command_contract():
    report = verify_before_fix(BASELINE)
    assert report == {"passed": True, "errors": [], "checked_file_count": 9}

