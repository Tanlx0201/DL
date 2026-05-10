"""Retriever: embed query -> search faiss -> enrich SQLite metadata."""

from __future__ import annotations

from app.models.database import DatabaseManager
from app.retrieval.embedder import Embedder
from app.retrieval.faiss_store import FAISSStore


class Searcher:
    """Trả kết quả có đầy đủ metadata chunk."""

    def __init__(self, db: DatabaseManager, embedder: Embedder, faiss_store: FAISSStore) -> None:
        self.db = db
        self.embedder = embedder
        self.faiss_store = faiss_store

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)
        hits = self.faiss_store.search(query_embedding, top_k)
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        chunk_rows = self.db.get_chunks_by_ids(chunk_ids)
        chunk_map = {int(row["id"]): row for row in chunk_rows}

        results: list[dict] = []
        for chunk_id, score in hits:
            row = chunk_map.get(chunk_id)
            if not row:
                continue
            results.append(
                {
                    "chunk_id": int(row["id"]),
                    "score": float(score),
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "page": int(row["page"]),
                    "chunk_index": int(row["chunk_index"]),
                    "text": row["text"],
                }
            )
        return results
