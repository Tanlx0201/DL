"""Quản lý cấu hình runtime bằng SQLite."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv


DEFAULT_CONFIG: dict[str, str] = {
    "openai_api_key": "",
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "ollama_base_url": "http://host.docker.internal:11434",
    "default_provider": "gemini",
    "default_model": "gemini-2.0-flash",
    "embedding_provider": "local",
    "top_k": "5",
    "chunk_size": "1000",
    "chunk_overlap": "200",
    "temperature": "0.1",
    "max_tokens": "2048",
}


class AppConfig:
    """Đọc/ghi cấu hình từ SQLite; .env chỉ dùng để seed lần đầu."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._ensure_table()
        self._seed_from_env_if_needed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.commit()

    def _seed_from_env_if_needed(self) -> None:
        load_dotenv(override=False)
        env_seed = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_CONFIG["ollama_base_url"]),
            "default_provider": os.getenv("DEFAULT_LLM_PROVIDER", DEFAULT_CONFIG["default_provider"]),
            "default_model": os.getenv("DEFAULT_LLM_MODEL", DEFAULT_CONFIG["default_model"]),
            "embedding_provider": os.getenv(
                "DEFAULT_EMBEDDING_PROVIDER", DEFAULT_CONFIG["embedding_provider"]
            ),
            "top_k": os.getenv("TOP_K", DEFAULT_CONFIG["top_k"]),
            "chunk_size": os.getenv("CHUNK_SIZE", DEFAULT_CONFIG["chunk_size"]),
            "chunk_overlap": os.getenv("CHUNK_OVERLAP", DEFAULT_CONFIG["chunk_overlap"]),
            "temperature": os.getenv("TEMPERATURE", DEFAULT_CONFIG["temperature"]),
            "max_tokens": os.getenv("MAX_TOKENS", DEFAULT_CONFIG["max_tokens"]),
        }

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT key, value FROM config").fetchall()
                existing = {row["key"] for row in rows}
                for key, fallback in DEFAULT_CONFIG.items():
                    value = env_seed.get(key, fallback) or fallback
                    if key not in existing:
                        conn.execute(
                            "INSERT INTO config(key, value) VALUES (?, ?)",
                            (key, value),
                        )
                conn.commit()

    def get(self, key: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        if row is None:
            return DEFAULT_CONFIG.get(key, "")
        return str(row["value"])

    def set(self, key: str, value: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO config(key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
                conn.commit()

    def bulk_set(self, payload: dict[str, str]) -> None:
        for key, value in payload.items():
            self.set(key, str(value))

    def get_llm_config(self) -> dict[str, str | int | float]:
        return {
            "provider": self.get("default_provider"),
            "model": self.get("default_model"),
            "temperature": float(self.get("temperature") or "0.1"),
            "max_tokens": int(self.get("max_tokens") or "2048"),
        }

    def get_embedding_config(self) -> dict[str, str]:
        return {
            "provider": self.get("embedding_provider"),
            "openai_api_key": self.get("openai_api_key"),
        }


def ensure_data_dirs(base_dir: str = "data") -> None:
    """Tạo thư mục dữ liệu cần thiết."""
    Path(base_dir, "faiss").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "uploads").mkdir(parents=True, exist_ok=True)
