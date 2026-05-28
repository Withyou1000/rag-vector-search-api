from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.main import create_app
from app.text_utils import tokenize


class HashEmbedding:
    """测试用 embedding，避免脚本依赖外部模型下载。"""

    dim = 16

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                vector[hash(token) % self.dim] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def build_test_client() -> TestClient:
    temp_root = Path(tempfile.mkdtemp(prefix="rag-smoke-"))
    app_settings = replace(
        settings,
        qdrant_path=str(temp_root / "qdrant"),
        data_dir=temp_root / "rag_data",
        documents_dir=temp_root / "rag_data" / "documents",
        metadata_path=temp_root / "rag_data" / "documents.json",
        traces_dir=temp_root / "rag_data" / "traces",
        enable_llm_answer=False,
        enable_query_rewrite=False,
        enable_multi_query=True,
    )
    client = TestClient(create_app(app_settings, embedding_model_override=HashEmbedding()))
    client.temp_root = temp_root  # type: ignore[attr-defined]
    return client


def main() -> None:
    client = build_test_client()
    try:
        sample_bytes = (PROJECT_ROOT / "samples" / "demo.md").read_bytes()
        response = client.post(
            "/documents",
            files={"file": ("demo.md", sample_bytes, "text/markdown")},
            data={
                "owner": "alice",
                "visibility": "private",
                "source": "smoke-test",
                "tags": "rag,embedding",
            },
        )
        response.raise_for_status()
        ingest = response.json()
        if ingest["chunk_count"] <= 0:
            raise AssertionError("上传后没有生成 chunk。")

        search = client.post(
            "/search",
            json={
                "query": "chunk 为什么不能太大",
                "top_k": 3,
                "mode": "hybrid",
                "enable_rerank": True,
                "user_id": "alice",
            },
        )
        search.raise_for_status()
        if not search.json()["hits"]:
            raise AssertionError("混合检索没有返回结果。")

        ask = client.post(
            "/ask",
            json={
                "question": "chunk 为什么不能太大？",
                "user_id": "alice",
                "mode": "hybrid",
                "enable_rerank": True,
            },
        )
        ask.raise_for_status()
        answer = ask.json()
        if not answer["citations"]:
            raise AssertionError("问答结果没有引用来源。")

        traces = client.get(f"/traces/{answer['retrieval_trace_id']}")
        traces.raise_for_status()
        print("smoke test ok")
    finally:
        shutil.rmtree(client.temp_root, ignore_errors=True)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
