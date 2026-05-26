from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import chunk_sections
from app.embeddings import HashingEmbedding, cosine_similarity
from app.parsers import parse_document
from app.vector_store import VectorStore


def main() -> None:
    """运行最小端到端验证，确认核心检索链路可以工作。"""

    sample_path = Path("samples/demo.md")
    sections = parse_document(sample_path.name, sample_path.read_bytes())
    chunks = chunk_sections(sections, chunk_size_tokens=80, overlap_tokens=15)
    if not chunks:
        raise AssertionError("示例文档没有切出 chunk。")

    embedding = HashingEmbedding(dim=128)
    query_vector, chunk_vector = embedding.embed(["chunk 为什么不能太大", chunks[0].text])
    similarity = cosine_similarity(query_vector, chunk_vector)
    if not -1.0 <= similarity <= 1.0:
        raise AssertionError("Cosine Similarity 超出合理范围。")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorStore(
            path=temp_dir,
            collection_name="smoke_chunks",
            embedding_model=embedding,
        )
        try:
            document_id = store.add_document(
                filename=sample_path.name,
                chunks=chunks,
                source="smoke-test",
                tags=["rag", "embedding"],
            )
            hits = store.search("chunk 太大会有什么问题", top_k=3)
            documents = store.list_documents()
        finally:
            # Windows 会锁住 Qdrant 的本地 SQLite 文件，关闭后临时目录才能被删除。
            store.close()

    # 这些断言覆盖了解析、切块、向量写入、搜索和文档聚合列表的主路径。
    if not document_id:
        raise AssertionError("没有生成 document_id。")
    if not hits:
        raise AssertionError("搜索没有返回任何结果。")
    if len(documents) != 1:
        raise AssertionError("文档列表聚合结果不正确。")

    print("smoke test ok")


if __name__ == "__main__":
    main()
