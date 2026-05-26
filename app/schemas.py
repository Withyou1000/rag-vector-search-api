from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None
    tags: list[str] | None = None


class SearchHit(BaseModel):
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source: str | None = None
    tags: list[str] = []
    chunk_count: int
