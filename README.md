# 第三阶段项目：Embedding 和向量检索

这是一个“个人知识库搜索 API”的最小完整项目，用来学习 RAG 的地基：解析文档、切块、生成 embedding、存入向量库、按 query 找回相关片段。

项目刻意不生成答案，只做搜索。这样你可以把注意力放在 RAG 最容易被低估的一步：资料到底有没有找准。

## 技术栈

| 组件 | 选择 |
| --- | --- |
| 后端 | Python FastAPI |
| 向量库 | Qdrant 本地模式 |
| 文档解析 | Markdown / txt 原生解析，PDF 使用 PyMuPDF |
| Embedding | 默认教学版 HashingEmbedding，可替换 sentence-transformers |
| 框架 | 裸写，不依赖 LangChain / LlamaIndex |

## 项目结构

```text
app/
  main.py          # FastAPI 接口
  parsers.py       # Markdown / txt / PDF 解析
  chunking.py      # 按段落、标题、代码块切块
  embeddings.py    # embedding 与 Cosine Similarity
  vector_store.py  # Qdrant 写入、检索、删除
  schemas.py       # API 输入输出模型
  config.py        # 环境变量配置
samples/
  demo.md          # 可上传的示例文档
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

也可以在 PowerShell 里直接运行：

```powershell
.\scripts\run_server.ps1
```

启动后访问：

- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## 基础验证

如果你只是想确认核心链路能跑通，可以执行：

```bash
python scripts/smoke_test.py
```

它会使用临时 Qdrant 目录完成一次“解析 → 切块 → embedding → 入库 → 查询”的端到端验证，不会污染 `.qdrant` 正式数据目录。

## 上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents" `
  -F "file=@samples/demo.md" `
  -F "source=learning" `
  -F "tags=rag,embedding"
```

返回示例：

```json
{
  "document_id": "生成的 UUID",
  "filename": "demo.md",
  "chunk_count": 2
}
```

## 搜索片段

```powershell
curl.exe -X POST "http://127.0.0.1:8000/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"为什么 chunk 不能太大？\",\"top_k\":3}"
```

返回的 `hits` 里包含：

- `score`：向量相似度分数
- `text`：召回到的原文片段
- `metadata`：文档来源、页码、标题、标签、chunk 序号等

## 可配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QDRANT_PATH` | `.qdrant` | Qdrant 本地数据目录 |
| `COLLECTION_NAME` | `knowledge_chunks` | collection 名称 |
| `EMBEDDING_PROVIDER` | `hash` | 默认离线教学 embedding |
| `EMBEDDING_DIM` | `384` | 哈希 embedding 向量维度 |
| `CHUNK_SIZE_TOKENS` | `350` | 每个 chunk 的粗略 token 上限 |
| `CHUNK_OVERLAP_TOKENS` | `60` | 相邻 chunk 重叠 token 数 |

## 换成真实语义 embedding

默认的 `HashingEmbedding` 不需要下载模型，适合先跑通流程。但它更接近关键词向量化，不是真正的语义理解。

如果你想体验真实语义检索，可以额外安装：

```bash
pip install sentence-transformers
set EMBEDDING_PROVIDER=sentence-transformers
set EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
uvicorn app.main:app --reload
```

注意：切换 embedding 维度后，旧的 Qdrant collection 维度可能不一致。学习阶段最简单的处理方式是停止服务后删除 `.qdrant` 目录，再重新上传文档。

## 学习路线

建议按这个顺序读代码：

1. `app/parsers.py`：不同格式怎么变成纯文本。
2. `app/chunking.py`：为什么按段落、标题和代码块切，而不是直接固定字符数。
3. `app/embeddings.py`：文本如何变成固定长度向量，Cosine Similarity 怎么算。
4. `app/vector_store.py`：向量和 metadata 如何写入 Qdrant，查询时如何用 `top_k` 和过滤条件。
5. `app/main.py`：API 如何把整条链路串起来。

## 下一步练习

- 增加 Hybrid Search：把关键词 BM25 与向量检索结果融合。
- 增加 Rerank：先召回 `top_k=20`，再用 reranker 选出最相关的 5 条。
- 增加权限过滤：在 metadata 中加入 `owner` 或 `visibility`。
- 增加网页前端：做一个上传文档和搜索结果页面。
