"""FAISS vector store chỉ lưu vectors + map chunk_ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import faiss
import numpy as np


class FAISSStore:
    """Quản lý IndexFlatIP và map vị trí -> chunk_id."""

    def __init__(self, index_dir: str, embedding_dim: int = 384) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.json"
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_ids: list[int] = []
        self.load()

    def _reset_index(self, dim: int) -> None:
        self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids = []

    def add(self, embeddings: np.ndarray, chunk_ids: list[int]) -> None:
        if embeddings.size == 0:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.embedding_dim:
            self._reset_index(int(embeddings.shape[1]))
        self.index.add(embeddings)
        self.chunk_ids.extend([int(cid) for cid in chunk_ids])

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0 or not self.chunk_ids:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def delete_by_doc_id(self, doc_id: str, all_chunks_callback: Callable[[str], tuple[np.ndarray, list[int]]]) -> None:
        all_embeddings, all_chunk_ids = all_chunks_callback(doc_id)
        self.rebuild(all_embeddings, all_chunk_ids)

    def rebuild(self, all_embeddings: np.ndarray | None, all_chunk_ids: list[int]) -> None:
        if all_embeddings is None or len(all_chunk_ids) == 0:
            self._reset_index(self.embedding_dim)
            return
        arr = np.asarray(all_embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._reset_index(int(arr.shape[1]))
        self.index.add(arr)
        self.chunk_ids = [int(cid) for cid in all_chunk_ids]

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps({"chunk_ids": self.chunk_ids}, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.embedding_dim = self.index.d
        if self.meta_path.exists():
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.chunk_ids = [int(cid) for cid in payload.get("chunk_ids", [])]

    def get_stats(self) -> dict[str, int]:
        return {
            "total_vectors": int(self.index.ntotal),
            "embedding_dim": int(self.embedding_dim),
        }
