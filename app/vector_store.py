from __future__ import annotations

from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.chunking import Chunk
from app.embeddings import EmbeddingModel
from app.storage import StoredDocument


class VectorStore:
    """封装 Qdrant 的 dense 检索与 chunk 读写，其他排序在应用层编排。"""

    def __init__(self, path: str, collection_name: str, embedding_model: EmbeddingModel):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(item.name == self.collection_name for item in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.dim,
                    distance=models.Distance.COSINE,
                ),
            )
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # 为权限、文档过滤与全文检索建立索引，后续过滤与定位更稳定。
        self.client.create_payload_index(
            self.collection_name,
            "document_id",
            models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            self.collection_name,
            "owner",
            models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            self.collection_name,
            "visibility",
            models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            self.collection_name,
            "filename",
            models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            self.collection_name,
            "tags",
            models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            self.collection_name,
            "text",
            models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=1,
            ),
        )

    def add_document(
        self,
        *,
        document: StoredDocument,
        chunks: list[Chunk],
    ) -> str:
        self.delete_document(document.document_id)
        vectors = self.embedding_model.embed(chunk.text for chunk in chunks)
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = str(uuid4())
            payload: dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": document.document_id,
                "filename": document.filename,
                "source": document.source,
                "source_path": document.source_path,
                "tags": document.tags,
                "owner": document.owner,
                "visibility": document.visibility,
                "document_version": document.version,
                "status": document.status,
                "updated_at": document.updated_at,
                "content_hash": document.content_hash,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page,
                "heading": chunk.heading,
                "token_estimate": chunk.token_count,
                "text": chunk.text,
            }
            points.append(models.PointStruct(id=chunk_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)
        return document.document_id

    def search(
        self,
        *,
        query: str,
        top_k: int,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedding_model.embed([query])[0]
        query_filter = self._build_filter(source=source, tags=tags, user_id=user_id)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [self._record_to_hit(item.payload or {}, item.score) for item in results]

    def list_chunks(
        self,
        *,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(source=source, tags=tags, user_id=user_id)
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [record.payload or {} for record in records]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def close(self) -> None:
        self.client.close()

    def _build_filter(
        self,
        *,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> models.Filter:
        must: list[Any] = [
            models.FieldCondition(key="status", match=models.MatchValue(value="active"))
        ]
        should = [
            models.FieldCondition(key="owner", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="visibility", match=models.MatchValue(value="shared")),
        ]
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        if tags:
            for tag in tags:
                must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=[tag])))
        return models.Filter(must=must, should=should, min_should=models.MinShould(conditions=should, min_count=1))

    @staticmethod
    def _record_to_hit(payload: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "score": float(score),
            "text": payload.get("text", ""),
            "metadata": {key: value for key, value in payload.items() if key != "text"},
        }
