from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
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


def build_client() -> TestClient:
    temp_root = Path(tempfile.mkdtemp(prefix="rag-eval-"))
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


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_hit_filenames(hits: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        filename = hit["metadata"]["filename"]
        if filename in seen:
            continue
        seen.add(filename)
        ordered.append(filename)
    return ordered


def seed_documents(client: TestClient) -> None:
    for filename in ("demo.md", "policy.md"):
        payload = (PROJECT_ROOT / "samples" / filename).read_bytes()
        response = client.post(
            "/documents",
            files={"file": (filename, payload, "text/markdown")},
            data={"owner": "alice", "visibility": "shared", "source": "evaluation"},
        )
        response.raise_for_status()


def recall_at_k(hits: list[dict], gold_filenames: set[str]) -> float:
    if not gold_filenames:
        return 1.0
    hit_docs = set(unique_hit_filenames(hits))
    return len(hit_docs & gold_filenames) / len(gold_filenames)


def mrr_at_k(hits: list[dict], gold_filenames: set[str]) -> float:
    for index, filename in enumerate(unique_hit_filenames(hits), start=1):
        if filename in gold_filenames:
            return 1.0 / index
    return 0.0


def ndcg_at_k(hits: list[dict], gold_filenames: set[str]) -> float:
    dcg = 0.0
    for index, filename in enumerate(unique_hit_filenames(hits), start=1):
        if filename in gold_filenames:
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, min(len(gold_filenames), len(unique_hit_filenames(hits))) + 1)
    )
    return dcg / ideal if ideal else 1.0


def evaluate(client: TestClient, dataset: list[dict]) -> dict[str, dict[str, float]]:
    modes = [
        ("vector", False),
        ("bm25", False),
        ("hybrid", False),
        ("hybrid", True),
    ]
    summary = defaultdict(
        lambda: {"Recall@k": 0.0, "MRR@k": 0.0, "NDCG@k": 0.0, "NoAnswerAcc": 0.0, "AnswerWithCitation": 0.0}
    )
    for sample in dataset:
        gold_filenames = set(sample.get("gold_filenames", []))
        for mode, enable_rerank in modes:
            label = "hybrid+rerank" if mode == "hybrid" and enable_rerank else mode
            search = client.post(
                "/search",
                json={
                    "query": sample["question"],
                    "top_k": 5,
                    "mode": mode,
                    "enable_rerank": enable_rerank,
                    "user_id": "alice",
                },
            )
            search.raise_for_status()
            hits = search.json()["hits"]
            summary[label]["Recall@k"] += recall_at_k(hits, gold_filenames)
            summary[label]["MRR@k"] += mrr_at_k(hits, gold_filenames)
            summary[label]["NDCG@k"] += ndcg_at_k(hits, gold_filenames)

            ask = client.post(
                "/ask",
                json={
                    "question": sample["question"],
                    "user_id": "alice",
                    "mode": mode,
                    "enable_rerank": enable_rerank,
                    "top_k": 5,
                },
            )
            ask.raise_for_status()
            answer = ask.json()
            should_answer = sample.get("should_answer", True)
            summary[label]["NoAnswerAcc"] += float(answer["no_answer"] == (not should_answer))
            summary[label]["AnswerWithCitation"] += float(bool(answer["citations"]))

    total = max(len(dataset), 1)
    for metrics in summary.values():
        for metric_name in list(metrics):
            metrics[metric_name] = round(metrics[metric_name] / total, 4)
    return dict(summary)


def main() -> None:
    dataset_path = PROJECT_ROOT / "samples" / "evaluation" / "company_eval.jsonl"
    dataset = load_dataset(dataset_path)
    client = build_client()
    try:
        seed_documents(client)
        report = evaluate(client, dataset)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(client.temp_root, ignore_errors=True)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
