"""Build local BM25 index from Stage1 chunk metadata."""

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
from rag_v2.retrieval.bm25_index import BM25Index


def build_bm25_index(
    metadata_path: Path,
    out_path: Path,
    report_path: Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    started = time.perf_counter()
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = BM25Index.from_chunk_metadata(rows, k1=k1, b=b)
    index.save(out_path)
    report = {
        "metadata_path": str(metadata_path),
        "index_path": str(out_path),
        "report_path": str(report_path),
        "doc_count": index.doc_count,
        "avg_doc_len": round(index.avg_doc_len, 2),
        "unique_terms": len(index.doc_freqs),
        "k1": k1,
        "b": b,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    cfg = get_config()
    stage3_dir = cfg.artifacts_dir / "stage3"
    metadata_path = args.metadata or (cfg.stage1_artifacts_dir / "chunk_metadata.json")
    out_path = args.out or (stage3_dir / "bm25_index.json")
    report_path = args.report or (stage3_dir / "bm25_build_report.json")
    result = build_bm25_index(metadata_path, out_path, report_path, k1=args.k1, b=args.b)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
