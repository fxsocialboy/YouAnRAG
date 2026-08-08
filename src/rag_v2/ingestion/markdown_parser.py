"""Lightweight Markdown parser for stage 1.

The parser intentionally avoids a full Markdown AST.  For this project we only
need a convergent, testable pass that keeps heading context for later chunking:
Markdown lines -> paragraph/list/table/reference/toc blocks with section_path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable

from rag_v2.ingestion.normalizer import normalize_markdown

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ORDERED_LIST_RE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)、]\s+|[（(]?[一二三四五六七八九十]+[）)]\s*|[①②③④⑤⑥⑦⑧⑨⑩]\s*)"
)
_TOC_DOTS_RE = re.compile(r"\.{2,}\s*\d+\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")
_REFERENCE_HEADING_RE = re.compile(r"^(参考文献|references?)\s*[:：]?\s*$", re.IGNORECASE)
_REFERENCE_ITEM_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)、])\s*.+")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(slots=True)
class MarkdownBlock:
    """A light block parsed from Markdown with inherited section context."""

    source_file: str
    section_path: list[str]
    text: str
    block_type: str
    is_indexable: bool
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_markdown_file(path: str | Path) -> list[MarkdownBlock]:
    """Read and parse a Markdown file using filename as fallback root heading."""

    md_path = Path(path)
    text = md_path.read_text(encoding="utf-8")
    return parse_markdown(text, source_file=md_path.name)


def parse_markdown(text: str, source_file: str) -> list[MarkdownBlock]:
    """Parse Markdown into lightweight blocks with a heading-stack section_path.

    - Markdown headings update a simple stack.
    - Consecutive non-heading lines of the same simple type are grouped.
    - Every emitted block carries the current heading path.
    - TOC/page-number/reference noise is marked ``is_indexable=False``.
    """

    normalized = normalize_markdown(text)
    root_title = _filename_title(source_file)
    heading_stack: list[str | None] = [root_title]
    blocks: list[MarkdownBlock] = []
    buffer: list[str] = []
    buffer_type: str | None = None
    buffer_start = 0
    in_reference_section = False

    def current_path() -> list[str]:
        return [item for item in heading_stack if item]

    def flush(end_line: int) -> None:
        nonlocal buffer, buffer_type, buffer_start
        if not buffer:
            return
        text_value = "\n".join(buffer).strip()
        if text_value:
            block_type = buffer_type or classify_line(text_value, in_reference_section)
            is_indexable = _is_indexable_text(text_value, block_type)
            blocks.append(
                MarkdownBlock(
                    source_file=source_file,
                    section_path=current_path(),
                    text=text_value,
                    block_type=block_type,
                    is_indexable=is_indexable,
                    start_line=buffer_start,
                    end_line=end_line,
                )
            )
        buffer = []
        buffer_type = None
        buffer_start = 0

    lines = normalized.split("\n") if normalized else []
    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            flush(line_no - 1)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush(line_no - 1)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = _update_heading_stack(heading_stack, level, title, root_title)
            in_reference_section = bool(_REFERENCE_HEADING_RE.match(title))
            blocks.append(
                MarkdownBlock(
                    source_file=source_file,
                    section_path=current_path(),
                    text=title,
                    block_type="heading",
                    is_indexable=False,
                    start_line=line_no,
                    end_line=line_no,
                    metadata={"heading_level": level},
                )
            )
            continue

        line_type = classify_line(line, in_reference_section)
        if buffer and line_type != buffer_type:
            flush(line_no - 1)
        if not buffer:
            buffer_start = line_no
            buffer_type = line_type
        buffer.append(line)

    flush(len(lines))
    return blocks


def classify_line(line: str, in_reference_section: bool = False) -> str:
    """Classify one normalized Markdown line into a small set of block types."""

    stripped = line.strip()
    if not stripped:
        return "blank"
    if _TABLE_LINE_RE.match(stripped):
        return "table"
    if in_reference_section or _REFERENCE_HEADING_RE.match(stripped) or _REFERENCE_ITEM_RE.match(stripped) and _looks_like_reference(stripped):
        return "reference"
    if _looks_like_toc(stripped):
        return "toc"
    if _ORDERED_LIST_RE.match(stripped):
        return "list"
    return "paragraph"


def _update_heading_stack(stack: list[str | None], level: int, title: str, root_title: str) -> list[str | None]:
    # Keep file title as stable root at position 0. Markdown H1 replaces root
    # when it is meaningful; lower levels are stored by heading level.
    new_stack = list(stack)
    if not new_stack:
        new_stack = [root_title]
    if level == 1:
        new_stack = [title]
    else:
        while len(new_stack) <= level - 1:
            new_stack.append(None)
        new_stack = new_stack[: level - 1]
        if not new_stack:
            new_stack = [root_title]
        new_stack.append(title)
    return new_stack


def _filename_title(source_file: str) -> str:
    return Path(source_file).stem or source_file


def _looks_like_toc(line: str) -> bool:
    if _PAGE_NUMBER_RE.match(line):
        return True
    if _TOC_DOTS_RE.search(line):
        return True
    # Common converted-PDF TOC lines such as "1.1 总体目标 15".
    return bool(re.match(r"^\s*\d+(?:\.\d+)*\s+.{2,40}\s+\d{1,4}\s*$", line))


def _looks_like_reference(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in ("doi", "journal", "press", "出版社", "学报", "vol.", "no."))


def _is_indexable_text(text: str, block_type: str) -> bool:
    if block_type in {"toc", "reference", "blank"}:
        return False
    if block_type == "heading":
        return False
    if len(text.strip()) <= 1:
        return False
    return True


def section_path_coverage(blocks: Iterable[MarkdownBlock]) -> float:
    """Return ratio of indexable blocks that have non-empty section_path."""

    indexable = [block for block in blocks if block.is_indexable]
    if not indexable:
        return 1.0
    with_path = [block for block in indexable if block.section_path]
    return len(with_path) / len(indexable)
