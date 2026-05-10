"""API quản trị cấu hình và thống kê."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.generation.llm_providers import PROVIDER_MODELS
from app.models.schemas import ConfigUpdateRequest

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/config")
def get_config(request: Request) -> dict:
    cfg = request.app.state.config
    keys = [
        "openai_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "ollama_base_url",
        "default_provider",
        "default_model",
        "embedding_provider",
        "top_k",
        "chunk_size",
        "chunk_overlap",
        "temperature",
        "max_tokens",
    ]
    return {
        "values": {k: cfg.get(k) for k in keys},
        "provider_models": PROVIDER_MODELS,
    }


@router.post("/config")
def update_config(payload: ConfigUpdateRequest, request: Request) -> dict:
    cfg = request.app.state.config
    cfg.bulk_set(payload.values)

    if "embedding_provider" in payload.values or "openai_api_key" in payload.values:
        request.app.state.embedder = request.app.state.create_embedder()
        request.app.state.searcher = request.app.state.create_searcher()
        request.app.state.pipeline.embedder = request.app.state.embedder

    return {"ok": True}


@router.get("/stats")
def get_stats(request: Request) -> dict:
    db = request.app.state.db
    faiss_store = request.app.state.faiss_store
    return {"db": db.get_stats(), "faiss": faiss_store.get_stats()}


@router.get("/logs")
def get_logs(request: Request) -> dict:
    db = request.app.state.db
    return {"items": db.list_query_logs()}
