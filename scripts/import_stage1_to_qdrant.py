"""Import Stage1 FAISS vectors and chunk metadata into Qdrant.

Stage2 deliberately reuses the embeddings produced in Stage1.  This avoids a
full BGE re-encoding pass and makes the Qdrant migration runnable on CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_v2.config import RagV2Config, get_config
from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig
from rag_v2.stores.vector_store import VectorPoint


DEFAULT_PAYLOAD_FIELDS = (
    "chunk_id",
    "source_file",
    "chunk_index",
    "section_path",
    "section_path_text",
    "content",
    "content_hash",
    "token_count",
    "char_count",
    "is_indexable",
)


def load_stage1_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunk metadata must be a JSON list")
    return [dict(item) for item in data]


def metadata_to_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only Qdrant payload fields needed by retrieval/filter/sync."""

    payload = {key: metadata.get(key) for key in DEFAULT_PAYLOAD_FIELDS if key in metadata}
    payload["is_active"] = True
    if "section_path_text" not in payload:
        section_path = payload.get("section_path") or []
        payload["section_path_text"] = " > ".join(str(item) for item in section_path)
    if "content_hash" not in payload and isinstance(metadata.get("metadata"), dict):
        content_hash = metadata["metadata"].get("content_hash")
        if content_hash:
            payload["content_hash"] = content_hash
    return payload


def iter_faiss_points(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    limit: int | None = None,
) -> Iterator[VectorPoint]:
    if index.ntotal != len(metadata):
        raise ValueError(f"index.ntotal={index.ntotal} != metadata_count={len(metadata)}")
    total = index.ntotal if limit is None else min(index.ntotal, max(0, limit))
    for i in range(total):
        item = metadata[i]
        chunk_id = str(item.get("chunk_id") or f"chunk-{i}")
        vector = index.reconstruct(i).astype("float32").tolist()
        yield VectorPoint(chunk_id=chunk_id, vector=vector, payload=metadata_to_payload(item))


def import_stage1_to_qdrant(
    *,
    index_path: str | Path,
    metadata_path: str | Path,
    qdrant_config: QdrantStoreConfig,
    report_path: str | Path,
    batch_size: int = 128,
    limit: int | None = None,
    recreate: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    started = time.perf_counter()
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    report_path = Path(report_path)

    index = faiss.read_index(str(index_path))
    metadata = load_stage1_metadata(metadata_path)
    if index.ntotal != len(metadata):
        raise ValueError(f"index.ntotal={index.ntotal} != metadata_count={len(metadata)}")

    effective_limit = index.ntotal if limit is None else min(index.ntotal, max(0, limit))
    store = QdrantStore(qdrant_config)
    try:
        if recreate and store.client.collection_exists(store.collection_name):
            store.client.delete_collection(collection_name=store.collection_name)
        store.ensure_collection()
        batch: list[VectorPoint] = []
        imported = 0
        for point in iter_faiss_points(index, metadata, limit=effective_limit):
            batch.append(point)
            if len(batch) >= batch_size:
                store.upsert_points(batch)
                imported += len(batch)
                batch.clear()
        if batch:
            store.upsert_points(batch)
            imported += len(batch)

        qdrant_count = store.count()
        health = store.health()
    finally:
        store.close()

    report = {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_path),
        "qdrant_mode": qdrant_config.mode,
        "qdrant_path": str(qdrant_config.path) if qdrant_config.path else None,
        "qdrant_url": qdrant_config.url,
        "collection": qdrant_config.collection_name,
        "vector_size": qdrant_config.vector_size,
        "faiss_ntotal": int(index.ntotal),
        "metadata_count": len(metadata),
        "import_limit": limit,
        "imported_points": imported,
        "qdrant_count": qdrant_count,
        "recreate": recreate,
        "batch_size": batch_size,
        "health": health,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_default_qdrant_config(cfg: RagV2Config, vector_size: int, collection: str | None = None) -> QdrantStoreConfig:
    qdrant_cfg = QdrantStoreConfig.from_rag_config(cfg, vector_size=vector_size)
    if collection:
        qdrant_cfg = QdrantStoreConfig(
            mode=qdrant_cfg.mode,
            path=qdrant_cfg.path,
            url=qdrant_cfg.url,
            collection_name=collection,
            vector_size=qdrant_cfg.vector_size,
            distance=qdrant_cfg.distance,
        )
    return qdrant_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Stage1 FAISS vectors into Qdrant Local")
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    index_path = args.index_path or cfg.stage1_artifacts_dir / "faiss_index.index"
    metadata_path = args.metadata_path or cfg.stage1_artifacts_dir / "chunk_metadata.json"
    report_path = args.report_path or cfg.stage2_artifacts_dir / "qdrant_import_report.json"

    index = faiss.read_index(str(index_path))
    qdrant_cfg = build_default_qdrant_config(cfg, vector_size=int(index.d), collection=args.collection)
    report = import_stage1_to_qdrant(
        index_path=index_path,
        metadata_path=metadata_path,
        qdrant_config=qdrant_cfg,
        report_path=report_path,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
