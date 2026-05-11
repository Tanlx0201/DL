"""Pydantic schemas cho API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChunkSchema(BaseModel):
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int


class DocumentSchema(BaseModel):
    doc_id: str
    name: str
    doc_type: Literal["pdf", "url"]
    source: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime


class SearchResultSchema(BaseModel):
    chunk_id: int
    score: float
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    citations: list[SearchResultSchema]
    latency_ms: int


class ConfigUpdateRequest(BaseModel):
    values: dict[str, str]


class URLAddRequest(BaseModel):
    urls: list[str]
