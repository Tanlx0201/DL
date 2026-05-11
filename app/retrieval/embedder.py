"""Embedding providers."""

from __future__ import annotations

import numpy as np


class EmbeddingError(Exception):
    """Lỗi tạo embedding."""


class Embedder:
    """Hỗ trợ OpenAI hoặc local sentence-transformers."""

    def __init__(self, provider: str, openai_api_key: str = "") -> None:
        self.provider = provider
        self.openai_api_key = openai_api_key
        self._local_model = None

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(model="text-embedding-3-small", input=texts)
            vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
            return self._normalize(vectors)
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("Tạo embedding OpenAI thất bại.") from exc

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        try:
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = self._local_model.encode(texts, convert_to_numpy=True)
            return self._normalize(np.asarray(vectors, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("Tạo embedding local thất bại.") from exc

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if self.provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise EmbeddingError("Câu hỏi rỗng, không thể embedding.")
        return self.embed_texts([query])[0]
