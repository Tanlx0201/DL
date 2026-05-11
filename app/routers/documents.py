"""API cho quản lý tài liệu."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.url_extractor import URLExtractionError
from app.models.schemas import URLAddRequest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/")
def list_documents(request: Request) -> dict:
    db = request.app.state.db
    return {"items": db.list_documents()}


@router.post("/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> dict:
    pipeline = request.app.state.pipeline
    created = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"File {file.filename} rỗng.")
        try:
            doc_id = pipeline.ingest_pdf(file.filename, content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Lỗi xử lý PDF {file.filename}: {exc}") from exc
        created.append({"doc_id": doc_id, "name": file.filename})
    return {"created": created}


@router.post("/urls")
def add_urls(payload: URLAddRequest, request: Request) -> dict:
    pipeline = request.app.state.pipeline
    created = []
    for url in payload.urls:
        url = url.strip()
        if not url:
            continue
        try:
            doc_id = pipeline.ingest_url(url)
        except URLExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Lỗi xử lý URL {url}: {exc}") from exc
        created.append({"doc_id": doc_id, "url": url})
    return {"created": created}


@router.delete("/{doc_id}")
def delete_document(doc_id: str, request: Request) -> dict:
    pipeline = request.app.state.pipeline
    pipeline.delete_document(doc_id)
    return {"ok": True}


@router.delete("/")
def clear_documents(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    pipeline.clear_all()
    return {"ok": True}


@router.post("/reindex")
def reindex_documents(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    pipeline.reindex()
    return {"ok": True}
