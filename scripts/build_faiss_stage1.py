"""Build FAISS index for stage-1 chunks.

Inputs:
  artifacts/stage1/chunk_metadata.json

Outputs:
  artifacts/stage1/faiss_index.index
  artifacts/stage1/faiss_build_report.json

This script never writes to legacy_snapshot/RAG.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.config import get_config
from rag_v2.embedding.bge_embedder import BGEEmbedder, BGEEmbedderConfig
from rag_v2.ingestion.metadata import write_json
from rag_v2.stores.faiss_store import FaissStore


def load_metadata(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage1_faiss(
    metadata_path: Path | None = None,
    index_path: Path | None = None,
    report_path: Path | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    limit: int = 0,
) -> dict:
    cfg = get_config()
    cfg.validate(require_model=True)
    stage_dir = cfg.stage1_artifacts_dir
    metadata_path = metadata_path or (stage_dir / "chunk_metadata.json")
    index_path = index_path or (stage_dir / "faiss_index.index")
    report_path = report_path or (stage_dir / "faiss_build_report.json")

    metadata = load_metadata(metadata_path)
    if limit > 0:
        metadata = metadata[:limit]
    if not metadata:
        raise ValueError("metadata is empty; run scripts/build_chunks_stage1.py first")

    texts = [row["embedding_text"] for row in metadata]
    started = time.perf_counter()
    embedder = BGEEmbedder(
        BGEEmbedderConfig(
            model_path=cfg.model_path,
            device=device,
            dtype="float32",
            batch_size=batch_size,
            max_length=512,
            use_query_instruction=cfg.use_query_instruction,
            query_instruction=cfg.query_instruction,
        )
    )
    vectors = embedder.encode_passages(texts)
    store = FaissStore.from_vectors(vectors, metadata)
    store.validate()
    store.save(index_path)
    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "metadata_path": str(metadata_path),
        "index_path": str(index_path),
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "index_ntotal": store.ntotal,
        "metadata_count": len(metadata),
        "model_path": str(cfg.model_path),
        "batch_size": batch_size,
        "device": device,
        "dtype": "float32",
        "max_length": 512,
        "passage_instruction": False,
        "use_query_instruction": cfg.use_query_instruction,
        "elapsed_seconds": elapsed,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Build only first N chunks for smoke test; 0 means all.")
    args = parser.parse_args()
    report = build_stage1_faiss(
        metadata_path=args.metadata_path,
        index_path=args.index_path,
        report_path=args.report_path,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
