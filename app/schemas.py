from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SearchMode = Literal["vector", "fulltext", "bm25", "hybrid"]
Visibility = Literal["private", "shared"]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    version: int
    owner: str
    visibility: Visibility
    created_new_version: bool


class ReindexResponse(BaseModel):
    document_id: str
    chunk_count: int
    version: int
    status: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None
    tags: list[str] | None = None
    mode: SearchMode = "vector"
    enable_rerank: bool = False
    user_id: str = Field(default="anonymous", min_length=1)


class SearchHit(BaseModel):
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    retrieval_mode: SearchMode
    hits: list[SearchHit]
    used_queries: list[str] = []


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page: int | None = None
    heading: str | None = None
    quote: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_id: str = Field(default="anonymous", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None
    tags: list[str] | None = None
    mode: SearchMode = "hybrid"
    enable_rerank: bool = True
    max_context_chunks: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_trace_id: str
    used_queries: list[str]
    no_answer: bool
    retrieval_trace: dict[str, Any]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source: str | None = None
    tags: list[str] = []
    owner: str
    visibility: Visibility
    version: int
    status: str
    updated_at: str
    content_hash: str
    chunk_count: int


class TraceResponse(BaseModel):
    trace_id: str
    created_at: str
    query: str
    mode: SearchMode
    top_k: int
    user_id: str
    used_queries: list[str]
    enable_rerank: bool
    retrieved_chunks: list[dict[str, Any]]
    final_context_chunks: list[dict[str, Any]]
    answer: str
    no_answer: bool
