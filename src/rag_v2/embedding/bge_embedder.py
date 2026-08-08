"""BGE embedding wrapper for stage 1.6.

The implementation keeps the legacy choice of using the CLS vector and L2
normalization, but fixes the instruction direction: passage embedding uses the
prepared embedding_text directly; query embedding can optionally prepend a query
instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(slots=True)
class BGEEmbedderConfig:
    model_path: str | Path
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 16
    max_length: int = 512
    use_query_instruction: bool = True
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    """Small BGE encoder wrapper used by build/search scripts."""

    def __init__(self, config: BGEEmbedderConfig):
        self.config = config
        model_path = str(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = _torch_dtype(config.dtype)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(config.device)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def prepare_query(self, query: str) -> str:
        query = query.strip()
        if self.config.use_query_instruction and self.config.query_instruction:
            return f"{self.config.query_instruction}{query}"
        return query

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(list(texts), add_query_instruction=False)

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        prepared = [self.prepare_query(query) for query in queries]
        return self.encode(prepared, add_query_instruction=False)

    def encode(self, texts: list[str], add_query_instruction: bool = False) -> np.ndarray:
        if add_query_instruction:
            texts = [self.prepare_query(text) for text in texts]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[np.ndarray] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
            ).to(self.model.device)
            with torch.no_grad():
                output = self.model(**inputs)
                cls_embedding = output.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
            vectors.append(normalized.detach().cpu().numpy().astype("float32"))
        return np.vstack(vectors)


def _torch_dtype(dtype: str):
    normalized = dtype.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def l2_normalize_np(vectors: np.ndarray) -> np.ndarray:
    """Normalize numpy vectors row-wise with zero-vector protection."""

    if vectors.size == 0:
        return vectors.astype("float32")
    vectors = vectors.astype("float32", copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
