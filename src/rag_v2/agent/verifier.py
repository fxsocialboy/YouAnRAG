"""Deterministic citation verification for Stage6 answers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from rag_v2.agent.models import Citation, EvidenceItem

CITATION_RE = re.compile(r"\[(S\d+)\]")
# Only policy-like exact facts need citation diagnostics.  Generic numbers such
# as list item ``1.`` or display terms such as ``4K`` are deliberately absent.
POLICY_FACT_RE = re.compile(
    r"(?:"
    r"(?:Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|I|II|III|IV|V|一|二|三|四|五)级(?:应急)?响应"
    r"|\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
    r"|第[一二三四五六七八九十百0-9]+条"
    r"|\d+(?:\.\d+)?\s*(?:%|％|人|户|万元|亿元|小时|分钟|毫米|级以上)"
    r")",
    re.IGNORECASE,
)
STRUCTURAL_LINE_RE = re.compile(r"^(?:#{1,6}\s+|[-*+]\s*(?:\*{0,2})?$|\d+[.、)]\s*$)")


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    used_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    unused_citations: list[str] = field(default_factory=list)
    has_key_fact_without_citation: bool = False
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "used_citations": self.used_citations,
            "missing_citations": self.missing_citations,
            "unused_citations": self.unused_citations,
            "has_key_fact_without_citation": self.has_key_fact_without_citation,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
        }


class CitationVerifier:
    """Rule-based verifier for answer citation integrity.

    Stage6 deliberately keeps this deterministic: it checks citation ids,
    evidence mapping and whether key fact lines carry at least one citation.
    Semantic support verification is left to the next stage.
    """

    def verify(self, answer: str, citations: list[Citation], evidence: list[EvidenceItem]) -> VerificationResult:
        citation_ids = {item.citation_id for item in citations}
        evidence_ids = {item.citation_id for item in evidence}
        used = _unique_in_order(CITATION_RE.findall(answer or ""))
        missing = [cid for cid in used if cid not in citation_ids]
        unused = [cid for cid in sorted(citation_ids, key=_citation_sort_key) if cid not in used]
        reasons: list[str] = []

        if missing:
            reasons.append("missing_citation")
        if citation_ids - evidence_ids:
            reasons.append("citation_without_evidence")
        diagnostics: list[str] = []
        if evidence_ids - citation_ids:
            diagnostics.append("evidence_without_citation")
        has_fact_without_citation = _has_key_fact_without_citation(answer or "")
        warnings = ["key_fact_without_citation"] if has_fact_without_citation else []

        return VerificationResult(
            passed=not reasons,
            reasons=reasons,
            used_citations=used,
            missing_citations=missing,
            unused_citations=unused,
            has_key_fact_without_citation=has_fact_without_citation,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def verify_with_repair(
        self,
        answer: str,
        citations: list[Citation],
        evidence: list[EvidenceItem],
    ) -> tuple[str, VerificationResult]:
        """Try one conservative repair pass using existing evidence markers.

        A marker is appended only when the exact policy-fact token occurs in an
        evidence chunk.  The method cannot create citations, evidence or facts.
        """

        initial = self.verify(answer, citations, evidence)
        if not initial.has_key_fact_without_citation:
            return answer, initial
        repaired = _repair_exact_fact_citations(answer, citations, evidence)
        return repaired, self.verify(repaired, citations, evidence)


def _has_key_fact_without_citation(answer: str) -> bool:
    for line in answer.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("[")
            or stripped.startswith("来源")
            or STRUCTURAL_LINE_RE.match(stripped)
            or _is_markdown_heading(stripped)
        ):
            continue
        if POLICY_FACT_RE.search(stripped) and not CITATION_RE.search(stripped):
            return True
    return False


def _is_markdown_heading(line: str) -> bool:
    if line.startswith("#"):
        return True
    # Bold-only lines are usually section labels rather than factual claims.
    return bool(re.fullmatch(r"\*\*[^*]+\*\*[:：]?", line))


def _repair_exact_fact_citations(
    answer: str,
    citations: list[Citation],
    evidence: list[EvidenceItem],
) -> str:
    valid_ids = {item.citation_id for item in citations}
    evidence_by_id = {item.citation_id: item for item in evidence if item.citation_id in valid_ids}
    repaired_lines: list[str] = []
    repaired_once = False
    for line in answer.splitlines():
        stripped = line.strip()
        match = POLICY_FACT_RE.search(stripped)
        if (
            repaired_once
            or not match
            or CITATION_RE.search(stripped)
            or STRUCTURAL_LINE_RE.match(stripped)
            or _is_markdown_heading(stripped)
        ):
            repaired_lines.append(line)
            continue
        token = match.group(0)
        supporting_id = next(
            (citation_id for citation_id, item in evidence_by_id.items() if token in item.content),
            None,
        )
        if supporting_id is None:
            repaired_lines.append(line)
            continue
        repaired_lines.append(f"{line.rstrip()} [{supporting_id}]")
        repaired_once = True
    return "\n".join(repaired_lines)


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _citation_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", value)
    return (int(match.group(1)), value) if match else (10**9, value)
