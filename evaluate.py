"""
Legacy RAG baseline evaluator for phase 0.

Example:
  python evaluate.py --top-k 10
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LEGACY_RAG = ROOT / "legacy_snapshot" / "RAG"
DEFAULT_MODEL = Path(r"G:\tiaozhanbei\Youan-AI-main\youan-multiagent\multi_agent_server\app\RAG\bge-large-zh-v1.5")


class LegacySearcher:
    def __init__(self, model_path: Path, index_path: Path, metadata_path: Path, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path), device_map=device, torch_dtype=torch.float32)
        self.model.eval()
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(metadata_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 10):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = torch.nn.functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1).cpu().numpy()
        distances, indices = self.index.search(embedding, top_k)
        return indices[0].tolist(), distances[0].tolist()


def load_queries(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=ROOT / "experiments" / "eval_queries.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-path", type=Path, default=LEGACY_RAG / "faiss_index.index")
    parser.add_argument("--metadata-path", type=Path, default=LEGACY_RAG / "chunk_metadata.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N queries; 0 means all queries.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "legacy_baseline_results.json")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.limit > 0:
        queries = queries[: args.limit]
    searcher = LegacySearcher(args.model_path, args.index_path, args.metadata_path, args.device)
    rows = []
    hit_at_5 = hit_at_10 = doc_hit_at_10 = 0
    started = time.perf_counter()

    for q in queries:
        t0 = time.perf_counter()
        ids, scores = searcher.search(q["query"], top_k=args.top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        hits = []
        for rank, (idx, score) in enumerate(zip(ids, scores), 1):
            chunk = searcher.chunks[idx]
            hits.append({
                "rank": rank,
                "global_index": idx,
                "score_l2": float(score),
                "source_file": chunk.get("source_file"),
                "chunk_index": chunk.get("chunk_index"),
                "preview": chunk.get("content", "")[:120],
            })
        target_file = q["relevant_source_file"]
        target_chunk = q["relevant_chunk_index"]
        chunk_hit_5 = any(h["source_file"] == target_file and h["chunk_index"] == target_chunk for h in hits[:5])
        chunk_hit_10 = any(h["source_file"] == target_file and h["chunk_index"] == target_chunk for h in hits[:10])
        doc_hit_10_flag = any(h["source_file"] == target_file for h in hits[:10])
        hit_at_5 += int(chunk_hit_5)
        hit_at_10 += int(chunk_hit_10)
        doc_hit_at_10 += int(doc_hit_10_flag)
        rows.append({**q, "latency_ms": latency_ms, "chunk_hit@5": chunk_hit_5, "chunk_hit@10": chunk_hit_10, "doc_hit@10": doc_hit_10_flag, "top10": hits[:10]})
        print(f"{q['id']} chunk@10={int(chunk_hit_10)} doc@10={int(doc_hit_10_flag)} latency_ms={latency_ms}", flush=True)

    n = len(queries) or 1
    summary = {
        "query_count": len(queries),
        "top_k": args.top_k,
        "chunk_recall@5": round(hit_at_5 / n, 4),
        "chunk_recall@10": round(hit_at_10 / n, 4),
        "doc_recall@10": round(doc_hit_at_10 / n, 4),
        "total_seconds": round(time.perf_counter() - started, 2),
        "model_path": str(args.model_path),
        "index_path": str(args.index_path),
        "metadata_path": str(args.metadata_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
