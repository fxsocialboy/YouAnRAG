# Stage1 vs Legacy Retrieval Report

## Stage1 Summary

```json
{
  "query_count": 25,
  "top_k": 10,
  "doc_recall@5": 0.96,
  "doc_recall@10": 0.96,
  "chunk_recall_exact_index@5": 0.12,
  "chunk_recall_exact_index@10": 0.12,
  "avg_latency_ms": 214.34,
  "p95_latency_ms": 248.89,
  "total_seconds": 5.36,
  "metric_note": "chunk_recall_exact_index uses legacy qrels and is strict diagnostic only after re-chunking; doc_recall is the primary comparable metric."
}
```

## Legacy Summary

```json
{
  "query_count": 25,
  "top_k": 10,
  "chunk_recall@5": 0.2,
  "chunk_recall@10": 0.2,
  "doc_recall@10": 0.96,
  "total_seconds": 4.42,
  "model_path": "G:\\tiaozhanbei\\Youan-AI-main\\youan-multiagent\\multi_agent_server\\app\\RAG\\bge-large-zh-v1.5",
  "index_path": "G:\\tiaozhanbei\\newrag\\legacy_snapshot\\RAG\\faiss_index.index",
  "metadata_path": "G:\\tiaozhanbei\\newrag\\legacy_snapshot\\RAG\\chunk_metadata.json"
}
```

## Comparable Metrics

| Metric | Legacy | Stage1 | Note |
|---|---:|---:|---|
| doc_recall@10 | 0.96 | 0.96 | primary comparable metric |
| chunk exact recall@10 | 0.2 | 0.12 | strict; qrels use legacy chunk_index |
| avg/p95 latency | N/A | 214.34 / 248.89 ms | includes query embedding |

## Interpretation

Stage1 uses a new chunking strategy, so legacy `chunk_index` labels are not directly equivalent to Stage1 `chunk_index`. 
For this phase, use document-level recall and qualitative Top-10 inspection as the main comparison. 
If strict chunk-level evaluation is needed later, create new qrels against Stage1 chunk IDs.
