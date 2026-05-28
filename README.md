# 第四阶段项目：公司内部知识库 RAG

这是一个面向学习与实战的可用型 RAG 项目，目标不是只做 demo，而是把一个内部知识库问答系统真正跑起来，并且能评估它什么时候答得不对。

当前版本支持：

- 上传 `PDF / Markdown / txt`
- 文档删除与重新索引
- 文档 `owner / visibility` 轻量权限隔离
- `vector / fulltext / bm25 / hybrid` 四种检索模式
- 可选 `rerank`
- 问答回答、引用来源、no-answer
- 记录每次检索到的 chunk 与最终上下文
- 简易 Web 页面
- 冒烟脚本、API 测试、离线评测脚本

## 技术栈

- 后端：`FastAPI`
- 向量库：`Qdrant` 本地模式
- Embedding：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 问答生成：`OpenAI 兼容 API`
- 文档解析：Markdown / txt 原生解析，PDF 使用 `PyMuPDF`

## 目录结构

```text
app/
  asgi.py         # Uvicorn 启动入口
  main.py         # FastAPI 应用工厂与路由
  parsers.py      # PDF / Markdown / txt 解析
  chunking.py     # 文档切块
  embeddings.py   # Dense embedding 模型
  vector_store.py # Qdrant dense 存取
  retrieval.py    # BM25 / fulltext / hybrid / rerank 编排
  llm.py          # Query rewrite 与答案生成
  storage.py      # 文档元数据与 retrieval trace 存储
  schemas.py      # API 模型
scripts/
  run_server.ps1  # 启动服务
  smoke_test.py   # 端到端冒烟验证
  evaluate_rag.py # 离线评测脚本
tests/
  test_rag_api.py # API 自动化测试
samples/
  demo.md
  policy.md
  evaluation/company_eval.jsonl
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
.\scripts\run_server.ps1
```

服务启动后访问：

- 文档与问答页面：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## 环境变量

`.env.example` 已给出常用配置，重点如下：

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
CHUNK_SIZE_TOKENS=350
CHUNK_OVERLAP_TOKENS=60
OPENAI_BASE_URL=
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
APP_DATA_DIR=.rag_data
```

说明：

- 如果不配置 `OPENAI_*`，系统仍可运行，但问答会退化成基于检索片段的摘要式回答。
- 本地 Qdrant 模式下会提示 payload index 在本地模式不生效，这是正常现象，不影响当前项目运行。

## API 概览

### 1. 上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents" `
  -F "file=@samples/demo.md" `
  -F "owner=alice" `
  -F "visibility=private" `
  -F "source=internal" `
  -F "tags=rag,embedding"
```

### 2. 搜索

```powershell
curl.exe -X POST "http://127.0.0.1:8000/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"chunk 为什么不能太大\",\"mode\":\"hybrid\",\"enable_rerank\":true,\"user_id\":\"alice\",\"top_k\":5}"
```

### 3. 问答

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"chunk 为什么不能太大？\",\"mode\":\"hybrid\",\"enable_rerank\":true,\"user_id\":\"alice\",\"top_k\":5}"
```

### 4. 重新索引

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/{document_id}/reindex"
```

### 5. 查看 retrieval trace

```powershell
curl.exe "http://127.0.0.1:8000/traces/{trace_id}"
```

## 检索模式说明

- `vector`：语义检索，适合表达不完全一致的问题
- `fulltext`：关键词命中，适合短语或明确术语
- `bm25`：词项相关性排序，适合文档型检索
- `hybrid`：融合 dense + lexical，默认推荐
- `rerank`：对候选集二次排序，通常更稳，但会增加延迟

## 评测

### 冒烟验证

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

### API 自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rag_api
```

### 离线评测

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

评测脚本会比较：

- `vector`
- `bm25`
- `hybrid`
- `hybrid + rerank`

并输出 `Recall@k`、`MRR@k`、`NDCG@k`、`NoAnswerAcc`、`AnswerWithCitation`。

## 当前实现说明

- Dense 检索由 Qdrant 承担。
- `fulltext / bm25 / hybrid / rerank` 在应用层统一编排，便于本地稳定运行与教学分析。
- 文档元数据、原文件与 retrieval trace 存在 `.rag_data/`。
- Freshness 当前按“最新 active version 生效”处理。

## 下一步建议

- 接入真正的 cross-encoder reranker
- 引入更系统的评测集与人工标注
- 增加文档级摘要、标签筛选和批量导入
- 加入登录鉴权与多租户权限模型
