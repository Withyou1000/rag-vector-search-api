from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class Settings:
    """集中管理项目配置，避免在各模块里散落读取环境变量。"""

    app_name: str = "Company Knowledge RAG"
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_chunks")
    qdrant_path: str = os.getenv("QDRANT_PATH", ".qdrant")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "350"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
    data_dir: Path = Path(os.getenv("APP_DATA_DIR", ".rag_data"))
    documents_dir: Path = Path(os.getenv("DOCUMENTS_DIR", ".rag_data/documents"))
    metadata_path: Path = Path(os.getenv("DOCUMENT_METADATA_PATH", ".rag_data/documents.json"))
    traces_dir: Path = Path(os.getenv("TRACES_DIR", ".rag_data/traces"))
    evaluation_dir: Path = Path(os.getenv("EVALUATION_DIR", "samples/evaluation"))
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_user_agent: str = os.getenv("OPENAI_USER_AGENT", "OpenAI/Python 1.0.0")
    openai_timeout_seconds: int = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    rerank_model: str | None = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)
    enable_query_rewrite: bool = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
    enable_multi_query: bool = os.getenv("ENABLE_MULTI_QUERY", "true").lower() == "true"
    enable_llm_answer: bool = os.getenv("ENABLE_LLM_ANSWER", "true").lower() == "true"
    answer_score_threshold: float = float(os.getenv("ANSWER_SCORE_THRESHOLD", "0.18"))
    min_answer_hits: int = int(os.getenv("MIN_ANSWER_HITS", "1"))
    default_search_limit: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
    default_candidate_limit: int = int(os.getenv("DEFAULT_CANDIDATE_LIMIT", "20"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
    max_context_chars_per_chunk: int = int(os.getenv("MAX_CONTEXT_CHARS_PER_CHUNK", "900"))

    def ensure_directories(self) -> None:
        """启动时提前创建本地数据目录，避免运行中到处判断。"""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
