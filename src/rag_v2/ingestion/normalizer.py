r"""Conservative Markdown text normalization for stage 1.

The legacy pipeline removed all whitespace with ``re.sub(r"\s+", "", text)``,
which glued English words together and destroyed Markdown structure.  This
module keeps structural newlines and word spaces while removing only low-value
noise.
"""

from __future__ import annotations

import html
import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
_TOO_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_markdown(text: str) -> str:
    """Normalize Markdown conservatively without destroying document structure.

    Rules intentionally kept small and deterministic:
    - normalize newlines to ``\n``;
    - decode HTML entities such as ``&amp;``;
    - apply Unicode NFC normalization;
    - remove invisible control characters;
    - compress horizontal spaces to one ordinary space;
    - preserve paragraph/list/heading newlines;
    - compress 3+ blank lines to 2 blank lines;
    - strip trailing spaces on each line and surrounding whitespace.
    """

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    # Use NFC instead of NFKC: NFKC would turn symbols such as "Ⅳ级" into
    # "IV级", which is less faithful to policy/emergency-plan source text.
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _TOO_MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_lines(text: str) -> list[str]:
    """Return normalized non-empty lines, preserving their order."""

    normalized = normalize_markdown(text)
    return [line.strip() for line in normalized.split("\n") if line.strip()]


def has_english_word_gluing(before: str, after: str) -> bool:
    """Heuristic used by tests/reports to catch accidental English word gluing.

    It returns True when a visible English word boundary from ``before`` appears
    to have been removed in ``after``.  This is not a full quality metric; it is
    a simple guardrail against reintroducing the legacy bug.
    """

    before_words = re.findall(r"[A-Za-z]+", before)
    if len(before_words) < 2:
        return False
    for left, right in zip(before_words, before_words[1:]):
        if f"{left} {right}" in before and f"{left}{right}" in after:
            return True
    return False

