from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """集中管理配置，方便之后替换 embedding 模型或向量库路径。"""

    app_name: str = "Personal Knowledge Search API"
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_chunks")
    qdrant_path: str = os.getenv("QDRANT_PATH", ".qdrant")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "350"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))


settings = Settings()
