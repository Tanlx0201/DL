"""FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import AppConfig, ensure_data_dirs
from app.generation.generator import Generator
from app.ingestion.pipeline import IngestionPipeline
from app.models.database import DatabaseManager
from app.retrieval.embedder import Embedder
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.searcher import Searcher
from app.routers import admin, chat, documents

ensure_data_dirs("data")

app = FastAPI(title="RAG Grading Assistant", version="1.0.0")


def create_embedder() -> Embedder:
    provider = app.state.config.get("embedding_provider")
    return Embedder(
        provider=provider,
        openai_api_key=app.state.config.get("openai_api_key"),
    )


def create_searcher() -> Searcher:
    return Searcher(app.state.db, app.state.embedder, app.state.faiss_store)


@app.on_event("startup")
def on_startup() -> None:
    app.state.db = DatabaseManager("data/database.db")
    app.state.config = AppConfig("data/database.db")
    app.state.faiss_store = FAISSStore("data/faiss")
    app.state.create_embedder = create_embedder
    app.state.create_searcher = create_searcher
    app.state.embedder = create_embedder()
    app.state.searcher = create_searcher()
    app.state.generator = Generator(app.state.config)
    app.state.pipeline = IngestionPipeline(
        db=app.state.db,
        config=app.state.config,
        embedder=app.state.embedder,
        faiss_store=app.state.faiss_store,
        upload_dir="data/uploads",
    )


@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok", "message": "RAG Grading Assistant đang chạy"}


app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)
