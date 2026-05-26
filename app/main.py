from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.chunking import chunk_sections
from app.config import settings
from app.embeddings import create_embedding_model
from app.parsers import parse_document
from app.schemas import DocumentInfo, IngestResponse, SearchHit, SearchRequest, SearchResponse
from app.vector_store import VectorStore


embedding_model = create_embedding_model(
    provider=settings.embedding_provider,
    dim=settings.embedding_dim,
    model_name=settings.embedding_model,
)
vector_store = VectorStore(
    path=settings.qdrant_path,
    collection_name=settings.collection_name,
    embedding_model=embedding_model,
)

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "embedding_provider": settings.embedding_provider}


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    tags: str | None = Form(default=None),
) -> IngestResponse:
    """上传文档并写入向量库，只负责检索资料，不生成答案。"""

    try:
        content = await file.read()
        sections = parse_document(file.filename or "uploaded.txt", content)
        chunks = chunk_sections(
            sections=sections,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        parsed_tags = _parse_tags(tags)
        document_id = vector_store.add_document(
            filename=file.filename or "uploaded.txt",
            chunks=chunks,
            source=source,
            tags=parsed_tags,
        )
    except (RuntimeError, ValueError) as exc:
        # 把解析、切块、依赖缺失等可预期问题转成清晰的 400，便于前端或调用方处理。
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        document_id=document_id,
        filename=file.filename or "uploaded.txt",
        chunk_count=len(chunks),
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """查询相关片段；这是 RAG 里的 Retrieve 阶段，不做 Generate。"""

    hits = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        source=request.source,
        tags=request.tags,
    )
    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        hits=[SearchHit(**hit) for hit in hits],
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**item) for item in vector_store.list_documents()]


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    vector_store.delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}


def _parse_tags(tags: str | None) -> list[str]:
    """把表单里的逗号分隔标签整理成列表，避免把空字符串写入 metadata。"""

    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]
