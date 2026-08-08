"""Token counting helpers for stage 1 chunking.

Stage 1.4 uses a lightweight regex tokenizer for deterministic unit tests and
fast development.  Stage 1.6 can switch to the real BGE tokenizer while keeping
this small interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def tokenize(self, text: str) -> list[str]: ...

    def detokenize(self, tokens: list[str]) -> str: ...


@dataclass(slots=True)
class RegexTokenCounter:
    """Small deterministic tokenizer.

    Chinese characters and punctuation are counted individually, while
    contiguous English letters/digits are counted as one token.  It is not meant
    to exactly match BGE tokenization, but it gives stable boundaries for stage
    1.4 tests and quality checks.
    """

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return _TOKEN_RE.findall(text)

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def detokenize(self, tokens: list[str]) -> str:
        # For chunk boundaries and tests, preserving every token is more
        # important than perfectly reconstructing original spacing.
        return "".join(tokens)


def split_by_token_limit(text: str, counter: TokenCounter, limit: int) -> list[str]:
    """Split text into pieces with at most ``limit`` tokens each."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    tokens = counter.tokenize(text)
    if not tokens:
        return []
    pieces: list[str] = []
    for start in range(0, len(tokens), limit):
        pieces.append(counter.detokenize(tokens[start : start + limit]))
    return pieces
