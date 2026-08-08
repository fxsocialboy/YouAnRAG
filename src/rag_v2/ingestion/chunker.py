"""Token-budget chunker for stage 1.

Input: lightweight MarkdownBlock objects from markdown_parser.
Output: Chunk objects with section_path, content and embedding_text.

The implementation is intentionally convergent: one fixed parameter set, no
parent/child hierarchy, no heavy AST, and no index writes.  It only produces
chunks for the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from rag_v2.config import Stage1ChunkParams
from rag_v2.ingestion.markdown_parser import MarkdownBlock
from rag_v2.ingestion.token_counter import RegexTokenCounter, TokenCounter, split_by_token_limit
from rag_v2.schemas import Chunk

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？；;.!?])")


@dataclass(slots=True)
class ChunkBuildStats:
    source_file: str
    input_blocks: int
    indexable_blocks: int
    output_chunks: int
    max_tokens: int
    missing_section_path_chunks: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "source_file": self.source_file,
            "input_blocks": self.input_blocks,
            "indexable_blocks": self.indexable_blocks,
            "output_chunks": self.output_chunks,
            "max_tokens": self.max_tokens,
            "missing_section_path_chunks": self.missing_section_path_chunks,
        }


def build_chunks(
    blocks: list[MarkdownBlock],
    params: Stage1ChunkParams | None = None,
    token_counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Build stage-1 chunks from parsed Markdown blocks."""

    params = params or Stage1ChunkParams()
    params.validate()
    counter = token_counter or RegexTokenCounter()

    chunks: list[Chunk] = []
    current_units: list[str] = []
    current_section: list[str] | None = None
    current_source: str | None = None

    def current_text() -> str:
        return "\n".join(unit for unit in current_units if unit).strip()

    def finalize_current() -> None:
        nonlocal current_units, current_section, current_source
        text = current_text()
        if not text:
            current_units = []
            return
        _append_chunk(chunks, source_file=current_source or "unknown.md", section_path=current_section or [], content=text, counter=counter)
        overlap = _make_overlap_text(text, counter, params.overlap_tokens)
        current_units = [overlap] if overlap else []

    for block in blocks:
        if not block.is_indexable or not block.text.strip():
            continue
        source_file = block.source_file
        section_path = block.section_path
        content_hard_max = _content_hard_max(source_file, section_path, counter, params.hard_max_tokens)
        units = _split_block_to_units(block.text, counter, content_hard_max)
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            # New source or section: close current chunk, but do not carry overlap
            # across section boundaries.
            if current_units and (source_file != current_source or section_path != current_section):
                text = current_text()
                if text:
                    _append_chunk(chunks, source_file=current_source or "unknown.md", section_path=current_section or [], content=text, counter=counter)
                current_units = []

            current_source = source_file
            current_section = list(section_path)

            if not current_units:
                current_units = [unit]
                # A single unit can still be at hard_max; finalize immediately
                # when it reaches soft_max to avoid oversized accumulation.
                if _embedding_count(source_file, section_path, current_text(), counter) >= params.soft_max_tokens:
                    finalize_current()
                continue

            candidate = current_text() + "\n" + unit
            candidate_tokens = _embedding_count(source_file, section_path, candidate, counter)
            if candidate_tokens <= params.target_tokens:
                current_units.append(unit)
            elif candidate_tokens <= params.soft_max_tokens and _should_keep_together(unit):
                current_units.append(unit)
            else:
                finalize_current()
                # If overlap + new unit would exceed hard_max, drop overlap.
                candidate_after_overlap = current_text() + "\n" + unit if current_text() else unit
                if _embedding_count(source_file, section_path, candidate_after_overlap, counter) > params.hard_max_tokens:
                    current_units = [unit]
                else:
                    current_units.append(unit)
                if _embedding_count(current_source or source_file, current_section or section_path, current_text(), counter) >= params.soft_max_tokens:
                    finalize_current()

    # Final tail: if too small, merge into previous same-section chunk when safe.
    tail = current_text()
    if tail:
        # Avoid emitting an overlap-only tail when the last real unit was
        # finalized at the end of input.
        if chunks and tail == _make_overlap_text(chunks[-1].content, counter, params.overlap_tokens):
            return _reindex_chunks(chunks, counter)
        if chunks and counter.count(tail) < params.min_tokens and chunks[-1].source_file == (current_source or "") and chunks[-1].section_path == (current_section or []):
            merged = chunks[-1].content + "\n" + tail
            if _embedding_count(chunks[-1].source_file, chunks[-1].section_path, merged, counter) <= params.hard_max_tokens:
                chunks[-1] = _make_chunk(
                    source_file=chunks[-1].source_file,
                    chunk_index=chunks[-1].chunk_index,
                    section_path=chunks[-1].section_path,
                    content=merged,
                    counter=counter,
                )
            else:
                _append_chunk(chunks, source_file=current_source or "unknown.md", section_path=current_section or [], content=tail, counter=counter)
        else:
            _append_chunk(chunks, source_file=current_source or "unknown.md", section_path=current_section or [], content=tail, counter=counter)

    # Re-index per source_file, because chunks may be built from multiple docs.
    return _reindex_chunks(chunks, counter)


def build_chunk_stats(source_file: str, blocks: list[MarkdownBlock], chunks: list[Chunk]) -> ChunkBuildStats:
    relevant_chunks = [chunk for chunk in chunks if chunk.source_file == source_file]
    return ChunkBuildStats(
        source_file=source_file,
        input_blocks=sum(1 for block in blocks if block.source_file == source_file),
        indexable_blocks=sum(1 for block in blocks if block.source_file == source_file and block.is_indexable),
        output_chunks=len(relevant_chunks),
        max_tokens=max((chunk.token_count for chunk in relevant_chunks), default=0),
        missing_section_path_chunks=sum(1 for chunk in relevant_chunks if not chunk.section_path),
    )


def _split_block_to_units(text: str, counter: TokenCounter, hard_max_tokens: int) -> list[str]:
    if counter.count(text) <= hard_max_tokens:
        return _split_by_sentence(text) or [text]

    units: list[str] = []
    for sentence in _split_by_sentence(text) or [text]:
        if counter.count(sentence) <= hard_max_tokens:
            units.append(sentence)
        else:
            units.extend(split_by_token_limit(sentence, counter, hard_max_tokens))
    return units


def _split_by_sentence(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts


def _should_keep_together(unit: str) -> bool:
    # Lists/tables/short clauses are usually better kept with adjacent text if
    # still under soft_max.
    return "\n" in unit or len(unit) < 200


def _make_overlap_text(text: str, counter: TokenCounter, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    tokens = counter.tokenize(text)
    if len(tokens) <= overlap_tokens:
        return text
    return counter.detokenize(tokens[-overlap_tokens:])


def _append_chunk(chunks: list[Chunk], source_file: str, section_path: list[str], content: str, counter: TokenCounter) -> None:
    if not content.strip():
        return
    source_existing = [chunk for chunk in chunks if chunk.source_file == source_file]
    chunk_index = len(source_existing)
    chunks.append(_make_chunk(source_file, chunk_index, section_path, content, counter))


def _make_chunk(source_file: str, chunk_index: int, section_path: list[str], content: str, counter: TokenCounter) -> Chunk:
    content = content.strip()
    embedding_text = make_embedding_text(source_file, section_path, content)
    return Chunk(
        chunk_id=f"{source_file}::{chunk_index}",
        source_file=source_file,
        chunk_index=chunk_index,
        section_path=list(section_path),
        content=content,
        embedding_text=embedding_text,
        token_count=counter.count(embedding_text),
        char_count=len(content),
        is_indexable=True,
        metadata={"section_path_text": " > ".join(section_path)},
    )


def _reindex_chunks(chunks: list[Chunk], counter: TokenCounter) -> list[Chunk]:
    counters: dict[str, int] = {}
    reindexed: list[Chunk] = []
    for chunk in chunks:
        idx = counters.get(chunk.source_file, 0)
        counters[chunk.source_file] = idx + 1
        reindexed.append(_make_chunk(chunk.source_file, idx, chunk.section_path, chunk.content, counter))
    return reindexed


def make_embedding_text(source_file: str, section_path: list[str], content: str) -> str:
    doc_name = Path(source_file).stem
    path = " > ".join(section_path) if section_path else doc_name
    return f"[文档] {doc_name}\n[章节] {path}\n[正文] {content}"


def _embedding_count(source_file: str, section_path: list[str], content: str, counter: TokenCounter) -> int:
    return counter.count(make_embedding_text(source_file, section_path, content))


def _content_hard_max(source_file: str, section_path: list[str], counter: TokenCounter, hard_max_tokens: int) -> int:
    prefix_tokens = counter.count(make_embedding_text(source_file, section_path, ""))
    # Keep a small but usable lower bound for pathological long file/section
    # names; later quality reports will surface such cases.
    return max(16, hard_max_tokens - prefix_tokens)
