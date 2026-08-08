"""Lightweight local BM25 index for Stage3 sparse retrieval.

Stage3 intentionally keeps sparse retrieval self-contained:
- no Elasticsearch
- no third-party BM25 package
- JSON artifact friendly
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path
from typing import Any


_ASCII_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(slots=True)
class BM25Document:
    chunk_id: str
    source_file: str
    chunk_index: int
    section_path_text: str
    content: str
    content_preview: str
    token_count: int
    tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BM25SearchHit:
    rank: int
    score: float
    chunk_id: str
    source_file: str
    chunk_index: int
    section_path_text: str
    content: str
    content_preview: str
    token_count: int
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BM25Index:
    """Compact BM25 implementation over Stage1 chunk metadata."""

    def __init__(
        self,
        documents: list[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not (0 <= b <= 1):
            raise ValueError("b must be between 0 and 1")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.avg_doc_len = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self._doc_id_to_index: dict[str, int] = {}
        self._build()

    @classmethod
    def from_chunk_metadata(
        cls,
        rows: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        docs = [metadata_to_bm25_document(row) for row in rows]
        return cls(docs, k1=k1, b=b)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = [BM25Document(**row) for row in data["documents"]]
        return cls(docs, k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "k1": self.k1,
            "b": self.b,
            "documents": [doc.to_dict() for doc in self.documents],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int = 10, source_file: str | None = None) -> list[BM25SearchHit]:
        query_terms = tokenize_text(query)
        if not query_terms or top_k <= 0 or not self.documents:
            return []

        scores: list[tuple[int, float, list[str]]] = []
        for idx, doc in enumerate(self.documents):
            if source_file and doc.source_file != source_file:
                continue
            matched_terms: list[str] = []
            score = self._score_doc(idx, query_terms, matched_terms)
            if score > 0:
                scores.append((idx, score, matched_terms))

        scores.sort(key=lambda item: (-item[1], self.documents[item[0]].chunk_id))
        hits: list[BM25SearchHit] = []
        for rank, (idx, score, matched_terms) in enumerate(scores[:top_k], 1):
            doc = self.documents[idx]
            hits.append(
                BM25SearchHit(
                    rank=rank,
                    score=round(score, 6),
                    chunk_id=doc.chunk_id,
                    source_file=doc.source_file,
                    chunk_index=doc.chunk_index,
                    section_path_text=doc.section_path_text,
                    content=doc.content,
                    content_preview=doc.content_preview,
                    token_count=doc.token_count,
                    matched_terms=matched_terms,
                )
            )
        return hits

    def _build(self) -> None:
        if not self.documents:
            self.avg_doc_len = 0.0
            return
        df_counter: defaultdict[str, int] = defaultdict(int)
        total_len = 0
        for idx, doc in enumerate(self.documents):
            if not doc.tokens:
                doc.tokens = tokenize_for_document(doc)
            tf = Counter(doc.tokens)
            self.doc_term_freqs.append(tf)
            doc_len = sum(tf.values())
            self.doc_lengths.append(doc_len)
            total_len += doc_len
            self._doc_id_to_index[doc.chunk_id] = idx
            for term in tf.keys():
                df_counter[term] += 1
        self.doc_freqs = dict(df_counter)
        self.avg_doc_len = total_len / len(self.documents) if self.documents else 0.0

    def _score_doc(self, idx: int, query_terms: list[str], matched_terms: list[str]) -> float:
        tf = self.doc_term_freqs[idx]
        doc_len = self.doc_lengths[idx] or 1
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            matched_terms.append(term)
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9))
            score += idf * numerator / denominator
        return score


def metadata_to_bm25_document(row: dict[str, Any]) -> BM25Document:
    content = str(row.get("content", ""))
    section_path_text = str(row.get("section_path_text") or row.get("metadata", {}).get("section_path_text", ""))
    source_file = str(row.get("source_file", ""))
    chunk_index = int(row.get("chunk_index", -1))
    chunk_id = str(row.get("chunk_id", f"{source_file}::{chunk_index}"))
    return BM25Document(
        chunk_id=chunk_id,
        source_file=source_file,
        chunk_index=chunk_index,
        section_path_text=section_path_text,
        content=content,
        content_preview=content[:160],
        token_count=int(row.get("token_count", 0)),
        tokens=tokenize_for_metadata_row(row),
    )


def tokenize_for_metadata_row(row: dict[str, Any]) -> list[str]:
    parts = [
        str(row.get("source_file", "")),
        str(row.get("section_path_text") or row.get("metadata", {}).get("section_path_text", "")),
        str(row.get("content", "")),
    ]
    return tokenize_text("\n".join(parts))


def tokenize_for_document(doc: BM25Document) -> list[str]:
    return tokenize_text("\n".join([doc.source_file, doc.section_path_text, doc.content]))


def tokenize_text(text: str) -> list[str]:
    """Lightweight tokenizer for Stage3 BM25.

    Strategy:
    - keep lowercase ascii words / numbers as tokens
    - keep each CJK char as one token
    - keep contiguous digit sequences
    """

    normalized = text.lower()
    tokens: list[str] = []
    i = 0
    while i < len(normalized):
        ch = normalized[i]
        if ch.isspace():
            i += 1
            continue
        ascii_match = _ASCII_WORD_RE.match(normalized, i)
        if ascii_match:
            tokens.append(ascii_match.group(0))
            i = ascii_match.end()
            continue
        if _CJK_CHAR_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < len(normalized) and normalized[j].isdigit():
                j += 1
            tokens.append(normalized[i:j])
            i = j
            continue
        i += 1
    return tokens
