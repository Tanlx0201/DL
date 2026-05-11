"""API hỏi đáp RAG."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask")
def ask(payload: ChatRequest, request: Request) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    config = request.app.state.config
    searcher = request.app.state.searcher
    generator = request.app.state.generator
    db = request.app.state.db

    top_k = int(config.get("top_k"))
    retrieved = searcher.search(query, top_k=top_k)
    if not retrieved:
        return {
            "answer": "Không tìm thấy đủ thông tin trong tài liệu.",
            "citations": [],
            "latency_ms": 0,
        }

    try:
        llm_response, latency_ms, provider, model = generator.answer(query, retrieved)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi sinh câu trả lời: {exc}") from exc

    db.log_query(
        query_text=query,
        retrieved_chunks=[
            {
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "source": item["source"],
                "page": item["page"],
            }
            for item in retrieved
        ],
        llm_provider=provider,
        llm_model=model,
        answer_text=llm_response.text,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        latency_ms=latency_ms,
    )

    return {
        "answer": llm_response.text,
        "citations": retrieved,
        "latency_ms": latency_ms,
    }
