from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class EmbeddingModel(Protocol):
    """embedding 模型的最小接口，后续可以无痛替换成真实语义模型。"""

    dim: int

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ...


@dataclass
class HashingEmbedding:
    """离线可用的教学版 embedding。

    它不是大模型语义 embedding，而是把 token 哈希到固定维度向量里。
    优点是没有下载模型和 API Key 门槛，适合先理解向量化、Cosine Similarity
    和 Vector DB 的完整链路。
    """

    dim: int = 384

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        # 批量接口和真实 embedding 服务保持一致，方便之后替换实现。
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = TOKEN_RE.findall(text.lower())
        vector = [0.0] * self.dim

        # 统计词频后再写入向量，避免重复词完全淹没长文档中的其他信息。
        for token, count in Counter(tokens).items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        # Qdrant 使用 Cosine 距离时也会处理归一化；这里显式归一化便于你学习公式。
        return [value / norm for value in vector]


class SentenceTransformerEmbedding:
    """可选的真实语义 embedding，本地安装 sentence-transformers 后启用。"""

    def __init__(self, model_name: str):
        # 延迟导入是为了让默认教学模式不需要安装大型模型依赖。
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "请先安装 sentence-transformers，或把 EMBEDDING_PROVIDER 改回 hash。"
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        # normalize_embeddings=True 让向量长度归一化，点积即可近似 Cosine Similarity。
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


def create_embedding_model(provider: str, dim: int, model_name: str) -> EmbeddingModel:
    """根据配置创建 embedding 模型，主流程不关心具体实现。"""

    if provider == "hash":
        return HashingEmbedding(dim=dim)
    if provider in {"sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbedding(model_name=model_name)
    raise ValueError(f"未知 EMBEDDING_PROVIDER: {provider}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """演示 Cosine Similarity 的核心公式，便于单独调试和学习。"""

    if len(left) != len(right):
        raise ValueError("两个向量维度必须一致，才能计算 Cosine Similarity。")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)
