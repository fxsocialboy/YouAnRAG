"""Stage2 manual Markdown-to-Qdrant synchronization script.

The first version is intentionally a script instead of a long-running service.
It supports scan-only change inspection and document-level sync for added,
modified, and deleted Markdown files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_v2.config import RagV2Config, get_config
from rag_v2.embedding.bge_embedder import BGEEmbedder, BGEEmbedderConfig
from rag_v2.ingestion.chunker import build_chunks
from rag_v2.ingestion.markdown_parser import parse_markdown_file
from rag_v2.ingestion.metadata import enrich_chunk_metadata
from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig, chunk_id_to_qdrant_point_id
from rag_v2.stores.vector_store import VectorPoint
from rag_v2.sync.file_tracker import FileChangeSet, FileSnapshot, detect_changes, scan_markdown_files
from rag_v2.sync.registry import ChunkMapping, DocumentRegistry


class PassageEmbedder(Protocol):
    def encode_passages(self, texts: list[str]) -> Any: ...


def filter_changes(changes: FileChangeSet, source_file: str | None = None) -> FileChangeSet:
    if not source_file:
        return changes
    return FileChangeSet(
        added=[item for item in changes.added if item.source_file == source_file],
        modified=[item for item in changes.modified if item.source_file == source_file],
        deleted=[item for item in changes.deleted if item == source_file],
        unchanged=[item for item in changes.unchanged if item.source_file == source_file],
    )


def scan_changes(cfg: RagV2Config, registry: DocumentRegistry, source_file: str | None = None) -> FileChangeSet:
    snapshots = scan_markdown_files(cfg.source_markdown_dir)
    if registry.db_path.exists():
        changes = detect_changes(registry, snapshots)
    else:
        # Keep --scan-only side-effect free when the registry has not been initialized yet.
        changes = FileChangeSet(added=snapshots)
    return filter_changes(changes, source_file=source_file)


def chunks_from_markdown(markdown_path: str | Path, cfg: RagV2Config) -> list[dict[str, Any]]:
    blocks = parse_markdown_file(markdown_path)
    chunks = build_chunks(blocks, params=cfg.chunk_params)
    return [enrich_chunk_metadata(chunk) for chunk in chunks if chunk.is_indexable]


def build_vector_points(chunk_metadata: list[dict[str, Any]], embedder: PassageEmbedder) -> list[VectorPoint]:
    texts = [str(item["embedding_text"]) for item in chunk_metadata]
    vectors = embedder.encode_passages(texts)
    if len(vectors) != len(chunk_metadata):
        raise ValueError(f"embedding count mismatch: {len(vectors)} != {len(chunk_metadata)}")
    points: list[VectorPoint] = []
    for item, vector in zip(chunk_metadata, vectors):
        payload = _sync_payload(item)
        points.append(VectorPoint(chunk_id=str(item["chunk_id"]), vector=vector.tolist(), payload=payload))
    return points


def sync_one_document(
    *,
    snapshot: FileSnapshot,
    cfg: RagV2Config,
    registry: DocumentRegistry,
    store: QdrantStore,
    embedder: PassageEmbedder,
    replace_existing: bool,
) -> dict[str, Any]:
    markdown_path = cfg.source_markdown_dir / snapshot.relative_path
    if replace_existing:
        store.delete_by_source_file(snapshot.source_file)

    chunk_metadata = chunks_from_markdown(markdown_path, cfg)
    points = build_vector_points(chunk_metadata, embedder)
    store.upsert_points(points)

    registry.upsert_document(
        source_file=snapshot.source_file,
        relative_path=snapshot.relative_path,
        content_hash=snapshot.content_hash,
        status="active",
        chunk_count=len(points),
    )
    registry.replace_chunk_mappings(
        snapshot.source_file,
        [
            ChunkMapping(
                chunk_id=str(item["chunk_id"]),
                source_file=snapshot.source_file,
                chunk_index=int(item["chunk_index"]),
                content_hash=str(item.get("content_hash", "")),
                qdrant_point_id=chunk_id_to_qdrant_point_id(str(item["chunk_id"])),
            )
            for item in chunk_metadata
        ],
    )
    return {"source_file": snapshot.source_file, "chunks": len(points), "replace_existing": replace_existing}


def apply_sync(
    *,
    cfg: RagV2Config,
    registry: DocumentRegistry,
    store: QdrantStore,
    embedder: PassageEmbedder,
    source_file: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    registry.init_schema()
    store.ensure_collection()
    changes = scan_changes(cfg, registry, source_file=source_file)

    added_reports = [
        sync_one_document(snapshot=item, cfg=cfg, registry=registry, store=store, embedder=embedder, replace_existing=False)
        for item in changes.added
    ]
    modified_reports = [
        sync_one_document(snapshot=item, cfg=cfg, registry=registry, store=store, embedder=embedder, replace_existing=True)
        for item in changes.modified
    ]

    deleted_reports: list[dict[str, Any]] = []
    for deleted_source in changes.deleted:
        store.delete_by_source_file(deleted_source)
        registry.mark_deleted(deleted_source)
        deleted_reports.append({"source_file": deleted_source, "deleted": True})

    return {
        "mode": "sync",
        "source_file_filter": source_file,
        "changes": changes.to_summary(),
        "added": added_reports,
        "modified": modified_reports,
        "deleted": deleted_reports,
        "qdrant_count": store.count(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def scan_only_report(cfg: RagV2Config, registry: DocumentRegistry, source_file: str | None = None) -> dict[str, Any]:
    changes = scan_changes(cfg, registry, source_file=source_file)
    return {
        "mode": "scan-only",
        "source_file_filter": source_file,
        "changes": changes.to_summary(),
        "added": [asdict(item) for item in changes.added],
        "modified": [asdict(item) for item in changes.modified],
        "deleted": changes.deleted,
        "unchanged": [item.source_file for item in changes.unchanged],
    }



class LazyBGEEmbedder:
    """Load the BGE model only when added/modified documents need encoding."""

    def __init__(self, cfg: RagV2Config, device: str, batch_size: int):
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self._embedder: BGEEmbedder | None = None

    def encode_passages(self, texts: list[str]) -> Any:
        if self._embedder is None:
            self._embedder = build_embedder(self.cfg, device=self.device, batch_size=self.batch_size)
        return self._embedder.encode_passages(texts)


def build_embedder(cfg: RagV2Config, device: str, batch_size: int) -> BGEEmbedder:
    return BGEEmbedder(
        BGEEmbedderConfig(
            model_path=cfg.model_path,
            device=device,
            batch_size=batch_size,
            use_query_instruction=cfg.use_query_instruction,
            query_instruction=cfg.query_instruction,
        )
    )


def _sync_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item["chunk_id"],
        "source_file": item["source_file"],
        "chunk_index": item["chunk_index"],
        "section_path": item.get("section_path", []),
        "section_path_text": item.get("section_path_text", ""),
        "content": item.get("content", ""),
        "content_hash": item.get("content_hash", ""),
        "token_count": item.get("token_count", 0),
        "char_count": item.get("char_count", 0),
        "is_indexable": item.get("is_indexable", True),
        "is_active": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 Markdown to Qdrant sync")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan-only", action="store_true", help="Only show detected changes; do not write SQLite or Qdrant")
    mode.add_argument("--sync", action="store_true", help="Apply added/modified/deleted Markdown changes")
    mode.add_argument("--rebuild-all", action="store_true", help="Reserved for full rebuild; intentionally disabled in Stage2.4")
    parser.add_argument("--source-file", default=None, help="Limit scan/sync to one Markdown filename, e.g. test_sync.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.rebuild_all:
        raise SystemExit("--rebuild-all is intentionally disabled in Stage2.4; use targeted --sync first")

    cfg = get_config()
    registry = DocumentRegistry(cfg.registry_db_path)
    if args.scan_only:
        report = scan_only_report(cfg, registry, source_file=args.source_file)
    else:
        qdrant_cfg = QdrantStoreConfig.from_rag_config(cfg, vector_size=1024)
        if args.collection:
            qdrant_cfg = QdrantStoreConfig(
                mode=qdrant_cfg.mode,
                path=qdrant_cfg.path,
                url=qdrant_cfg.url,
                collection_name=args.collection,
                vector_size=qdrant_cfg.vector_size,
                distance=qdrant_cfg.distance,
            )
        store = QdrantStore(qdrant_cfg)
        try:
            embedder = LazyBGEEmbedder(cfg, device=args.device, batch_size=args.batch_size)
            report = apply_sync(
                cfg=cfg,
                registry=registry,
                store=store,
                embedder=embedder,
                source_file=args.source_file,
            )
        finally:
            store.close()

    report_path = args.report_path or cfg.stage2_artifacts_dir / ("sync_scan_report.json" if args.scan_only else "sync_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
