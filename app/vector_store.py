from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.chunking import Chunk
from app.embeddings import EmbeddingModel


class VectorStore:
    """封装 Qdrant 操作，让 API 层只关心“存 chunk”和“搜 chunk”。"""

    def __init__(self, path: str, collection_name: str, embedding_model: EmbeddingModel):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(item.name == self.collection_name for item in collections)
        if exists:
            return

        # Cosine 距离适合衡量方向相近程度，是文本 embedding 检索最常用的距离之一。
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_model.dim,
                distance=models.Distance.COSINE,
            ),
        )

    def add_document(
        self,
        filename: str,
        chunks: list[Chunk],
        source: str | None,
        tags: list[str],
    ) -> str:
        document_id = str(uuid4())
        vectors = self.embedding_model.embed(chunk.text for chunk in chunks)

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid4())
            payload: dict[str, Any] = {
                "document_id": document_id,
                "filename": filename,
                "source": source,
                "tags": tags,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page,
                "heading": chunk.heading,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        # upsert 支持重复执行写入；生产环境可换成更严格的幂等 document_id。
        self.client.upsert(collection_name=self.collection_name, points=points)
        return document_id

    def search(
        self,
        query: str,
        top_k: int,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedding_model.embed([query])[0]
        query_filter = self._build_filter(source=source, tags=tags)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for item in results:
            payload = item.payload or {}
            hits.append(
                {
                    "score": item.score,
                    "text": payload.get("text", ""),
                    "metadata": {key: value for key, value in payload.items() if key != "text"},
                }
            )
        return hits

    def list_documents(self) -> list[dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"chunk_count": 0, "tags": []}
        )
        for record in records:
            payload = record.payload or {}
            document_id = payload.get("document_id")
            if not document_id:
                continue

            # Qdrant 存的是 chunk，列表接口需要按 document_id 聚合成文档视角。
            item = grouped[document_id]
            item["document_id"] = document_id
            item["filename"] = payload.get("filename")
            item["source"] = payload.get("source")
            item["tags"] = payload.get("tags") or []
            item["chunk_count"] += 1
        return list(grouped.values())

    def delete_document(self, document_id: str) -> None:
        # 删除时按 payload 过滤，避免 API 层知道每个 chunk 的 point id。
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
        """释放 Qdrant 本地文件句柄，测试或脚本退出时尤其需要。"""

        self.client.close()

    def _build_filter(
        self,
        source: str | None,
        tags: list[str] | None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if source:
            conditions.append(
                models.FieldCondition(key="source", match=models.MatchValue(value=source))
            )
        if tags:
            for tag in tags:
                conditions.append(
                    models.FieldCondition(key="tags", match=models.MatchAny(any=[tag]))
                )

        if not conditions:
            return None
        return models.Filter(must=conditions)
