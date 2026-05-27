from dataclasses import dataclass
import os

from dotenv import load_dotenv


# 自动加载项目根目录的 .env，让配置不依赖手动设置 PowerShell 环境变量。
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class Settings:
    """集中管理配置，方便统一查看和调整。"""

    app_name: str = "Personal Knowledge Search API"
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_chunks")
    qdrant_path: str = os.getenv("QDRANT_PATH", ".qdrant")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "350"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))


settings = Settings()
