# 第三阶段项目：Embedding 和向量检索

这是一个“个人知识库搜索 API”的最小完整项目，用来学习 RAG 的检索地基：解析文档、切块、生成 embedding、写入向量库、按 query 召回相关片段。

项目刻意不生成答案，只做搜索。这样你可以先把注意力放在最关键的一步：资料能不能找准。

## 技术栈

| 组件 | 选择 |
| --- | --- |
| 后端 | Python FastAPI |
| 向量库 | Qdrant 本地模式 |
| 文档解析 | Markdown / txt 原生解析，PDF 使用 PyMuPDF |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 框架 | 裸写，不依赖 LangChain / LlamaIndex |

## 项目结构

```text
app/
  main.py          # FastAPI 接口
  parsers.py       # Markdown / txt / PDF 解析
  chunking.py      # 按段落、标题、代码块切块
  embeddings.py    # sentence-transformers embedding
  vector_store.py  # Qdrant 写入、检索、删除
  schemas.py       # API 输入输出模型
  config.py        # 配置加载
samples/
  demo.md          # 可上传的示例文档
scripts/
  run_server.ps1   # Windows PowerShell 启动脚本
  smoke_test.py    # 基础烟测
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
.\scripts\run_server.ps1
```

也可以直接用普通命令启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

启动后访问：

- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

默认启动后就会使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
项目根目录提供了一个可直接编辑的 `.env`，如果你后面想换模型或调整切块参数，直接改 `.env` 即可。

## 首次运行说明

- 第一次运行真实语义 embedding 时，会联网下载 Hugging Face 模型到本地缓存，首次启动会明显慢一些。
- 如果你之前已经建过旧的向量库，改模型后要先删除 `.qdrant`，再重新上传文档。

清理旧库示例：

```powershell
Remove-Item -Recurse -Force .qdrant
```

## 基础验证

如果你想先确认核心链路能跑通，可以执行：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

它会用临时 Qdrant 目录完成一次“解析 -> 切块 -> embedding -> 入库 -> 查询”的端到端验证，不会污染 `.qdrant` 正式数据目录。

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
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 默认多语言 embedding 模型 |
| `EMBEDDING_DIM` | `384` | 当前模型输出向量维度 |
| `CHUNK_SIZE_TOKENS` | `350` | 每个 chunk 的粗略 token 上限 |
| `CHUNK_OVERLAP_TOKENS` | `60` | 相邻 chunk 重叠 token 数 |

这些默认值现在已经写在项目根目录的 `.env` 里。

## 学习路线

建议按这个顺序读代码：

1. `app/parsers.py`：不同格式怎么变成纯文本。
2. `app/chunking.py`：为什么按段落、标题和代码块切，而不是直接固定字符数。
3. `app/embeddings.py`：文本如何变成固定长度向量。
4. `app/vector_store.py`：向量和 metadata 如何写入 Qdrant，查询时如何用 `top_k` 和过滤条件。
5. `app/main.py`：API 如何把整条链路串起来。

## 下一步练习

- 增加 Hybrid Search：把关键词 BM25 与向量检索结果融合。
- 增加 Rerank：先召回 `top_k=20`，再用 reranker 选出最相关的 5 条。
- 增加权限过滤：在 metadata 中加入 `owner` 或 `visibility`。
- 增加网页前端：做一个上传文档和搜索结果页面。
