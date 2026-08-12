"""Best-effort, project-relative Legacy snapshot integrity report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "legacy_code_hashes.json"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "stage7" / "after_fix" / "legacy_hash_report.json"


def _resolve_legacy_path(raw: str) -> Path | None:
    normalized = raw.replace("\\", "/")
    marker = "legacy_snapshot/RAG/"
    if marker in normalized:
        relative = normalized[normalized.index(marker):]
        return PROJECT_ROOT / Path(relative)
    # 阶段0清单曾记录原Windows绝对路径。隔离项目只允许校验同名的
    # legacy_snapshot/RAG文件，避免AutoDL/Linux把合法快照误判为越界路径。
    original_marker = "multi_agent_server/app/RAG/"
    if original_marker in normalized:
        name = normalized.rsplit("/", 1)[-1]
        return PROJECT_ROOT / "legacy_snapshot" / "RAG" / name
    return None


def check(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        path = _resolve_legacy_path(str(item.get("path", "")))
        if path is None:
            rows.append({"manifest_path": item.get("path"), "status": "outside_legacy_snapshot"})
            continue
        if not path.exists():
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "status": "missing"})
            continue
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "status": "ok" if current == item.get("sha256") else "hash_changed",
            "expected_sha256": item.get("sha256"), "actual_sha256": current,
        })
    return {"passed": all(row["status"] == "ok" for row in rows), "items": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        report = check(args.manifest)
    except Exception as exc:
        report = {"passed": False, "error": {"type": type(exc).__name__, "message": str(exc)}, "items": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Integrity failure is recorded but deliberately does not prevent packaging
    # already completed expensive experiments.


if __name__ == "__main__":
    main()
