from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class EmbeddingModel(Protocol):
    """embedding 模型的最小接口。"""

    dim: int

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ...


@dataclass
class SentenceTransformerEmbedding:
    """基于 sentence-transformers 的真实语义 embedding。"""

    model_name: str

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("请先安装 sentence-transformers。") from exc

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            # 模型首次下载后，受限网络环境里可能卡在 Hub 元数据探测。
            # 这里退回只读本地缓存，让已经下载好的模型仍然能离线使用。
            self.model = SentenceTransformer(self.model_name, local_files_only=True)

        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


def create_embedding_model(model_name: str) -> EmbeddingModel:
    """创建项目默认使用的 sentence-transformers embedding 模型。"""

    return SentenceTransformerEmbedding(model_name=model_name)
