from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.config import settings
from app.llm import AnswerGenerator
from app.main import create_app
from app.retrieval import RetrievalService
from app.storage import TraceRepository
from app.text_utils import tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HashEmbedding:
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


class RagApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="rag-test-"))
        app_settings = replace(
            settings,
            qdrant_path=str(self.temp_root / "qdrant"),
            data_dir=self.temp_root / "rag_data",
            documents_dir=self.temp_root / "rag_data" / "documents",
            metadata_path=self.temp_root / "rag_data" / "documents.json",
            traces_dir=self.temp_root / "rag_data" / "traces",
            enable_llm_answer=False,
            enable_query_rewrite=False,
            enable_multi_query=True,
        )
        self.client = TestClient(
            create_app(app_settings, embedding_model_override=HashEmbedding())
        )
        self.demo_bytes = (PROJECT_ROOT / "samples" / "demo.md").read_bytes()
        self.policy_bytes = (PROJECT_ROOT / "samples" / "policy.md").read_bytes()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _upload(self, filename: str, content: bytes, owner: str, visibility: str) -> dict:
        response = self.client.post(
            "/documents",
            files={"file": (filename, content, "text/markdown")},
            data={"owner": owner, "visibility": visibility, "source": "tests"},
        )
        response.raise_for_status()
        return response.json()

    def test_private_document_only_visible_to_owner(self) -> None:
        self._upload("demo.md", self.demo_bytes, owner="alice", visibility="private")
        alice_hits = self.client.post(
            "/search",
            json={"query": "chunk 为什么不能太大", "mode": "hybrid", "user_id": "alice"},
        ).json()["hits"]
        bob_hits = self.client.post(
            "/search",
            json={"query": "chunk 为什么不能太大", "mode": "hybrid", "user_id": "bob"},
        ).json()["hits"]
        self.assertTrue(alice_hits)
        self.assertFalse(bob_hits)

    def test_shared_document_visible_to_other_user(self) -> None:
        self._upload("policy.md", self.policy_bytes, owner="alice", visibility="shared")
        bob_hits = self.client.post(
            "/search",
            json={"query": "敏感文档默认是什么可见性", "mode": "bm25", "user_id": "bob"},
        ).json()["hits"]
        self.assertTrue(bob_hits)

    def test_ask_returns_citation_and_trace(self) -> None:
        self._upload("policy.md", self.policy_bytes, owner="alice", visibility="shared")
        response = self.client.post(
            "/ask",
            json={"question": "敏感文档默认是什么可见性？", "user_id": "alice", "mode": "hybrid"},
        )
        response.raise_for_status()
        payload = response.json()
        self.assertTrue(payload["citations"])
        trace = self.client.get(f"/traces/{payload['retrieval_trace_id']}")
        trace.raise_for_status()
        self.assertIn("retrieved_chunks", trace.json())

    def test_delete_and_reindex_document(self) -> None:
        ingest = self._upload("demo.md", self.demo_bytes, owner="alice", visibility="private")
        reindex = self.client.post(f"/documents/{ingest['document_id']}/reindex")
        reindex.raise_for_status()
        delete_response = self.client.delete(f"/documents/{ingest['document_id']}")
        delete_response.raise_for_status()
        hits = self.client.post(
            "/search",
            json={"query": "chunk 为什么不能太大", "mode": "vector", "user_id": "alice"},
        ).json()["hits"]
        self.assertFalse(hits)

    def test_no_answer_when_corpus_has_no_support(self) -> None:
        self._upload("policy.md", self.policy_bytes, owner="alice", visibility="shared")
        response = self.client.post(
            "/ask",
            json={"question": "员工报销流程是什么？", "user_id": "alice", "mode": "hybrid"},
        )
        response.raise_for_status()
        self.assertTrue(response.json()["no_answer"])

    def test_hybrid_hits_are_not_blocked_by_rrf_threshold_when_llm_is_available(self) -> None:
        llm_client = Mock()
        llm_client.enabled = True
        service = RetrievalService(
            settings=settings,
            vector_store=Mock(),
            document_repository=Mock(),
            trace_repository=TraceRepository(self.temp_root / "trace-check"),
            query_planner=Mock(),
            answer_generator=AnswerGenerator(
                llm_client=llm_client,
                enable_llm_answer=True,
            ),
        )
        hits = [
            {
                "score": 0.03,
                "text": "蒋佳城 移动端开发 全栈开发 2005年6月",
                "metadata": {"chunk_id": "chunk-1"},
            }
        ]

        self.assertFalse(
            service._is_low_confidence(
                question="这个简历是谁的",
                mode="hybrid",
                hits=hits,
            )
        )


if __name__ == "__main__":
    unittest.main()
