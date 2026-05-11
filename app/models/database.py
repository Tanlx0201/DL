"""SQLite data access layer."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class DatabaseManager:
    """Thao tác với SQLite cho documents, chunks, query logs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT NOT NULL,
                    retrieved_chunks TEXT NOT NULL,
                    llm_provider TEXT,
                    llm_model TEXT,
                    answer_text TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    latency_ms INTEGER
                );
                """
            )
            conn.commit()

    def create_document(
        self,
        name: str,
        doc_type: str,
        source: str,
        page_count: int,
        chunk_count: int,
        doc_id: str | None = None,
    ) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, name, doc_type, source, page_count, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, name, doc_type, source, page_count, chunk_count),
            )
            conn.commit()
        return doc_id

    def update_document_chunk_count(self, doc_id: str, chunk_count: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE documents SET chunk_count = ? WHERE doc_id = ?", (chunk_count, doc_id))
            conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, name, doc_type, source, page_count, chunk_count, uploaded_at
                FROM documents ORDER BY uploaded_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def clear_documents(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.commit()

    def insert_chunks(self, chunks: list[dict[str, Any]], embeddings: np.ndarray | None = None) -> list[int]:
        if not chunks:
            return []
        ids: list[int] = []
        with self._connect() as conn:
            cursor = conn.cursor()
            for idx, chunk in enumerate(chunks):
                emb_blob = None
                if embeddings is not None:
                    emb_blob = np.asarray(embeddings[idx], dtype=np.float32).tobytes()
                cursor.execute(
                    """
                    INSERT INTO chunks(doc_id, source, page, chunk_index, text, char_start, char_end, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["doc_id"],
                        chunk["source"],
                        int(chunk["page"]),
                        int(chunk["chunk_index"]),
                        chunk["text"],
                        int(chunk["char_start"]),
                        int(chunk["char_end"]),
                        emb_blob,
                    ),
                )
                ids.append(int(cursor.lastrowid))
            conn.commit()
        return ids

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
                chunk_ids,
            ).fetchall()
        row_map = {int(r["id"]): dict(r) for r in rows}
        return [row_map[cid] for cid in chunk_ids if cid in row_map]

    def get_all_chunk_embeddings(self) -> tuple[np.ndarray, list[int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id ASC"
            ).fetchall()
        chunk_ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            chunk_ids.append(int(row["id"]))
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32), []
        return np.vstack(vectors).astype(np.float32), chunk_ids

    def update_chunk_embedding(self, chunk_id: int, embedding: np.ndarray) -> None:
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._connect() as conn:
            conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (blob, chunk_id))
            conn.commit()

    def get_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            query_count = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        return {
            "total_documents": int(doc_count),
            "total_chunks": int(chunk_count),
            "total_queries": int(query_count),
        }

    def log_query(
        self,
        query_text: str,
        retrieved_chunks: list[dict[str, Any]],
        llm_provider: str,
        llm_model: str,
        answer_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs(
                    query_text, retrieved_chunks, llm_provider, llm_model,
                    answer_text, prompt_tokens, completion_tokens, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_text,
                    json.dumps(retrieved_chunks, ensure_ascii=False),
                    llm_provider,
                    llm_model,
                    answer_text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                ),
            )
            conn.commit()

    def list_query_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["retrieved_chunks"] = json.loads(item["retrieved_chunks"])
            result.append(item)
        return result
