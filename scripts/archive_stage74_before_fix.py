"""Archive and verify the immutable Stage 7.4 before-fix baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT.parent / "stage7_results.tar.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "stage7" / "before_fix"

RESULT_SPECS = {
    "final_labeled_legacy_eval.json": 25,
    "final_labeled_v2_eval.json": 25,
    "final_random_v2_eval.json": 120,
}
SUPPORT_FILES = (
    "final_metrics.json",
    "final_report.md",
    "stage7_manual_review.md",
)
LOG_FILES = (
    "logs/01_labeled_legacy.log",
    "logs/02_labeled_v2.log",
    "logs/03_random_v2.log",
)
FROZEN_DATASETS = (
    "experiments/eval_queries_final_labeled.jsonl",
    "experiments/eval_queries_final_random.jsonl",
)
CODE_SNAPSHOT_FILES = (
    "evaluate_stage7.py",
    "src/rag_v2/config.py",
    "src/rag_v2/evaluation/models.py",
    "src/rag_v2/agent/service.py",
    "src/rag_v2/agent/guardrail.py",
    "src/rag_v2/agent/verifier.py",
    "src/rag_v2/agent/llm_composer.py",
    "src/rag_v2/retrieval/multi_query_pipeline.py",
    "src/rag_v2/retrieval/rerank_pipeline.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_member(members: dict[str, tarfile.TarInfo], relative: str) -> tarfile.TarInfo:
    suffix = "/artifacts/stage7/" + relative
    matches = [item for name, item in members.items() if name.endswith(suffix) or name == "artifacts/stage7/" + relative]
    if len(matches) != 1 or not matches[0].isfile():
        raise FileNotFoundError(f"archive must contain exactly one regular file for {relative!r}")
    return matches[0]


def _copy_member(bundle: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    stream = bundle.extractfile(member)
    if stream is None:
        raise OSError(f"cannot read archive member {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(stream.read())


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_clean": status == "" if status is not None else None,
    }


def _validate_result(path: Path, expected_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if payload.get("query_count") != expected_count or len(rows) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} rows")
    bad = [row.get("query_id") for row in rows if row.get("status") not in {"ok", "partial"}]
    if bad:
        raise ValueError(f"{path.name}: non-success rows: {bad}")
    return {
        "query_count": expected_count,
        "ok_count": sum(row.get("status") == "ok" for row in rows),
        "partial_count": sum(row.get("status") == "partial" for row in rows),
        "config": payload.get("config", {}),
        "environment": payload.get("environment", {}),
    }


def archive_before_fix(source_archive: Path = DEFAULT_ARCHIVE, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    wanted = [*RESULT_SPECS, *SUPPORT_FILES, *LOG_FILES]
    with tarfile.open(source_archive, "r:gz") as bundle:
        members = {PurePosixPath(item.name).as_posix(): item for item in bundle.getmembers()}
        for relative in wanted:
            _copy_member(bundle, _find_member(members, relative), output_dir / relative)

    result_snapshots = {
        name: _validate_result(output_dir / name, expected)
        for name, expected in RESULT_SPECS.items()
    }
    archived_files = {
        relative: {
            "sha256": sha256_file(output_dir / relative),
            "size_bytes": (output_dir / relative).stat().st_size,
        }
        for relative in wanted
    }
    frozen_datasets = {
        relative: {"sha256": sha256_file(PROJECT_ROOT / relative), "size_bytes": (PROJECT_ROOT / relative).stat().st_size}
        for relative in FROZEN_DATASETS
    }
    code_snapshot = {
        relative: {"sha256": sha256_file(PROJECT_ROOT / relative), "size_bytes": (PROJECT_ROOT / relative).stat().st_size}
        for relative in CODE_SNAPSHOT_FILES
        if (PROJECT_ROOT / relative).is_file()
    }
    model_configs: dict[str, Any] = {}
    for name in ("bge-large-zh-v1.5", "bge-reranker-base"):
        config = PROJECT_ROOT / "models" / name / "config.json"
        model_configs[name] = (
            {"config_path": str(config.relative_to(PROJECT_ROOT)), "config_sha256": sha256_file(config)}
            if config.is_file()
            else {"config_path": None, "config_sha256": None}
        )
    manifest = {
        "schema_version": 1,
        "stage": "7.4.1",
        "baseline": "before_fix",
        "source_archive": {"name": source_archive.name, "sha256": sha256_file(source_archive)},
        "results": result_snapshots,
        "archived_files": archived_files,
        "frozen_datasets": frozen_datasets,
        "code_snapshot": code_snapshot,
        "git": _git_snapshot(),
        "model_configs": model_configs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "before_fix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_before_fix(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest_path = output_dir / "before_fix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in manifest["archived_files"].items():
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            errors.append(f"archived file hash mismatch: {relative}")
    for relative, expected in manifest["frozen_datasets"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            errors.append(f"frozen dataset hash mismatch: {relative}")
    for name, expected_count in RESULT_SPECS.items():
        try:
            _validate_result(output_dir / name, expected_count)
        except Exception as exc:
            errors.append(str(exc))
    return {"passed": not errors, "errors": errors, "checked_file_count": len(manifest["archived_files"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_before_fix(args.output_dir) if args.verify_only else archive_before_fix(args.archive, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.verify_only and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
