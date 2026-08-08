"""Stage3 metadata filter planning.

Phase 3.3 deliberately activates only source_file hard filtering.
Region / hazard / document-type hooks are reserved for later metadata labeling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


_NON_WORD_RE = re.compile(r"[\s\-_—–·•,，。；：:（）()《》“”\"'`]+")


@dataclass(slots=True)
class MetadataFilterPlan:
    source_file: str | None = None
    region_hint: str | None = None
    hazard_hint: str | None = None
    document_type_hint: str | None = None
    reserved_fields: dict[str, str | None] = field(default_factory=dict)
    active_filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_source_file_catalog(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        source_file = str(row.get("source_file", "")).strip()
        if source_file and source_file not in seen:
            seen.add(source_file)
            ordered.append(source_file)
    return ordered


def detect_metadata_filters(query: str, source_files: list[str]) -> MetadataFilterPlan:
    query = query.strip()
    matched_source_file = detect_source_file_filter(query, source_files)
    reserved = {
        "region": detect_region_hint(query),
        "hazard": detect_hazard_hint(query),
        "document_type": detect_document_type_hint(query),
    }
    active_filters: dict[str, Any] = {}
    if matched_source_file:
        active_filters["source_file"] = matched_source_file
    return MetadataFilterPlan(
        source_file=matched_source_file,
        region_hint=reserved["region"],
        hazard_hint=reserved["hazard"],
        document_type_hint=reserved["document_type"],
        reserved_fields=reserved,
        active_filters=active_filters,
    )


def detect_source_file_filter(query: str, source_files: list[str]) -> str | None:
    normalized_query = normalize_text_for_match(query)
    if not normalized_query:
        return None
    candidates: list[tuple[int, str]] = []
    for source_file in source_files:
        name = source_file.strip()
        if not name:
            continue
        normalized_name = normalize_text_for_match(name)
        stem = name.rsplit(".", 1)[0]
        normalized_stem = normalize_text_for_match(stem)
        if normalized_query == normalized_name or normalized_query == normalized_stem:
            candidates.append((len(name), name))
            continue
        if normalized_name and normalized_name in normalized_query:
            candidates.append((len(name), name))
            continue
        if normalized_stem and normalized_stem in normalized_query:
            candidates.append((len(name), name))
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def detect_region_hint(query: str) -> str | None:
    for token in ["深圳", "广东", "河南", "湖北", "四川", "陕西", "北京", "天津", "西藏", "吉林", "云南"]:
        if token in query:
            return token
    return None


def detect_hazard_hint(query: str) -> str | None:
    for token in ["台风", "暴雨", "洪涝", "地震", "山洪", "地质灾害", "气象灾害"]:
        if token in query:
            return token
    return None


def detect_document_type_hint(query: str) -> str | None:
    for token in ["预案", "规划", "报告", "公报", "通知", "规范", "办法"]:
        if token in query:
            return token
    return None


def normalize_text_for_match(text: str) -> str:
    lowered = text.lower().strip()
    lowered = _NON_WORD_RE.sub("", lowered)
    return lowered
