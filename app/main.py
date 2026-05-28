from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.chunking import chunk_sections
from app.config import Settings, settings
from app.embeddings import EmbeddingModel, create_embedding_model
from app.llm import AnswerGenerator, OpenAICompatibleClient, QueryPlanner
from app.parsers import parse_document
from app.retrieval import RetrievalService
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    IngestResponse,
    ReindexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TraceResponse,
)
from app.storage import DocumentRepository, TraceRepository
from app.vector_store import VectorStore


def create_app(
    app_settings: Settings = settings,
    *,
    embedding_model_override: EmbeddingModel | None = None,
) -> FastAPI:
    """创建并返回 FastAPI 应用实例。

    这里使用“应用工厂（application factory）”模式，而不是在模块导入时立刻把所有对象写死。
    这样做的好处是：
    1. 测试时可以传入临时配置和假的 embedding 模型。
    2. 正式运行时仍然可以走默认 settings。
    3. 后面如果要接不同环境配置，也更容易扩展。
    """

    # ensure_directories 会提前创建运行需要的目录，
    # 避免后面真正写文件时才发现目录不存在。
    app_settings.ensure_directories()

    # embedding_model_override 是给测试用的可选覆盖项。
    # Python 里的“or”不只是布尔运算，也常用来写“有值就用它，没有就退回默认值”。
    embedding_model = embedding_model_override or create_embedding_model(
        model_name=app_settings.embedding_model
    )

    # 下面这些对象是整个应用的核心依赖：
    # - VectorStore: 负责向量存储与 chunk 读取
    # - DocumentRepository: 保存文档元数据
    # - TraceRepository: 保存检索追踪
    # - OpenAICompatibleClient: 调用兼容 LLM 接口
    vector_store = VectorStore(
        path=app_settings.qdrant_path,
        collection_name=app_settings.collection_name,
        embedding_model=embedding_model,
    )
    document_repository = DocumentRepository(
        metadata_path=app_settings.metadata_path,
        documents_dir=app_settings.documents_dir,
    )
    trace_repository = TraceRepository(traces_dir=app_settings.traces_dir)
    llm_client = OpenAICompatibleClient(app_settings)

    # RetrievalService 把检索、重排、上下文压缩、no-answer 判断都收在一起，
    # 路由层只要把请求参数传进去，不需要知道太多内部细节。
    retrieval_service = RetrievalService(
        settings=app_settings,
        vector_store=vector_store,
        document_repository=document_repository,
        trace_repository=trace_repository,
        query_planner=QueryPlanner(
            llm_client=llm_client,
            enable_query_rewrite=app_settings.enable_query_rewrite,
            enable_multi_query=app_settings.enable_multi_query,
        ),
        answer_generator=AnswerGenerator(
            llm_client=llm_client,
            enable_llm_answer=app_settings.enable_llm_answer,
        ),
    )

    app = FastAPI(title=app_settings.app_name)

    # app.state 是 FastAPI/Starlette 提供的一块“应用级共享存储”。
    # 它适合放整个服务启动后长期复用的对象，而不是每个请求都重新创建。
    app.state.settings = app_settings
    app.state.vector_store = vector_store
    app.state.document_repository = document_repository
    app.state.retrieval_service = retrieval_service
    app.state.trace_repository = trace_repository
    app.state.llm_client = llm_client

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        # 根路径直接返回内嵌 HTML，方便这个练手项目不额外引入前端框架。
        return _load_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        # llm_last_error 会把最近一次 LLM 调用失败原因暴露出来，
        # 这样前端看起来像 fallback 时，至少后端能快速定位原因。
        return {
            "status": "ok",
            "embedding_model": app_settings.embedding_model,
            "llm_enabled": str(llm_client.enabled).lower(),
            "llm_last_error": llm_client.last_error or "",
        }

    @app.post("/documents", response_model=IngestResponse)
    async def ingest_document(
        file: UploadFile = File(...),
        source: str | None = Form(default=None),
        tags: str | None = Form(default=None),
        owner: str = Form(default="anonymous"),
        visibility: str = Form(default="private"),
        reindex_if_exists: bool = Form(default=False),
    ) -> IngestResponse:
        """上传文档并建立索引。

        这里用了 async def，因为 UploadFile.read() 是异步接口。
        也就是说，这个函数内部可以使用 await file.read()。
        """

        try:
            content = await file.read()
            filename = file.filename or "uploaded.txt"
            parsed_tags = _parse_tags(tags)

            # parse_document 负责把 PDF / Markdown / txt 统一转成 section 列表。
            sections = parse_document(filename, content)

            # chunk_sections 会按照配置的大小和 overlap 切块。
            chunks = chunk_sections(
                sections=sections,
                chunk_size_tokens=app_settings.chunk_size_tokens,
                overlap_tokens=app_settings.chunk_overlap_tokens,
            )

            # save_upload 会负责处理版本、自定义 owner/visibility、幂等判断和原文件落盘。
            stored_document, created_new_version = document_repository.save_upload(
                filename=filename,
                content=content,
                source=source,
                tags=parsed_tags,
                owner=owner,
                visibility=visibility,
                chunk_count=len(chunks),
                reindex_if_exists=reindex_if_exists,
            )

            if created_new_version:
                # 只有真正产生新版本时，才重新写向量库。
                vector_store.add_document(document=stored_document, chunks=chunks)
            else:
                # 如果只是命中了同内容旧文档，这里同步一下 chunk_count，保持接口返回一致。
                stored_document = document_repository.update_chunk_count(
                    stored_document.document_id,
                    len(chunks),
                )
        except (RuntimeError, ValueError) as exc:
            # raise ... from exc 是“异常链”语法：
            # 既保留外层给前端看的 HTTP 400，也保留底层原始异常，方便排查。
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return IngestResponse(
            document_id=stored_document.document_id,
            filename=stored_document.filename,
            chunk_count=stored_document.chunk_count,
            version=stored_document.version,
            owner=stored_document.owner,
            visibility=stored_document.visibility,
            created_new_version=created_new_version,
        )

    @app.post("/documents/{document_id}/reindex", response_model=ReindexResponse)
    def reindex_document(document_id: str) -> ReindexResponse:
        """按文档 ID 重新解析并重建索引。"""

        document = document_repository.bump_version_for_reindex(document_id)

        # Path(...).read_bytes() 的写法表示：
        # 1. 先把字符串路径包成 Path 对象
        # 2. 再直接按二进制读取文件内容
        sections = parse_document(document.filename, Path(document.source_path).read_bytes())
        chunks = chunk_sections(
            sections=sections,
            chunk_size_tokens=app_settings.chunk_size_tokens,
            overlap_tokens=app_settings.chunk_overlap_tokens,
        )
        document_repository.update_chunk_count(document.document_id, len(chunks))
        vector_store.add_document(document=document, chunks=chunks)
        return ReindexResponse(
            document_id=document.document_id,
            chunk_count=len(chunks),
            version=document.version,
            status="reindexed",
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """返回检索命中的 chunk，不做最终答案生成。"""

        try:
            hits, trace_payload = retrieval_service.search(
                query=request.query,
                mode=request.mode,
                top_k=request.top_k,
                source=request.source,
                tags=request.tags,
                user_id=request.user_id,
                enable_rerank=request.enable_rerank,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # [SearchHit(**hit) for hit in hits] 是列表推导式：
        # 它等价于“遍历 hits，把每个字典都转成 SearchHit 对象，再收集成新列表”。
        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            retrieval_mode=request.mode,
            hits=[SearchHit(**hit) for hit in hits],
            used_queries=trace_payload["used_queries"],
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """执行完整问答链路：检索、上下文压缩、答案生成、引用返回。"""

        result = retrieval_service.ask(
            question=request.question,
            mode=request.mode,
            top_k=request.top_k,
            source=request.source,
            tags=request.tags,
            user_id=request.user_id,
            enable_rerank=request.enable_rerank,
            max_context_chunks=request.max_context_chunks,
        )
        return AskResponse(**result)

    @app.get("/documents", response_model=list[DocumentInfo])
    def list_documents(user_id: str | None = None) -> list[DocumentInfo]:
        """返回当前用户可见的文档列表。"""

        documents = document_repository.list_documents(owner=user_id)
        # document.__dict__ 是对象实例内部属性字典，
        # 这里直接拿来喂给 Pydantic 模型，省掉手动逐个字段复制。
        return [DocumentInfo(**document.__dict__) for document in documents]

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, str]:
        """删除文档元数据并从向量库移除对应 chunk。"""

        document_repository.mark_deleted(document_id)
        vector_store.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}

    @app.get("/traces/{trace_id}", response_model=TraceResponse)
    def get_trace(trace_id: str) -> TraceResponse:
        """查看某次问答的检索 trace。"""

        trace = retrieval_service.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return TraceResponse(**trace)

    return app


def _parse_tags(tags: str | None) -> list[str]:
    """把表单里的逗号分隔标签清理成列表。"""

    if not tags:
        return []

    # [tag.strip() for tag in tags.split(",") if tag.strip()] 是列表推导式：
    # 1. tags.split(",") 先按逗号切开
    # 2. tag.strip() 去掉每个标签两侧空白
    # 3. if tag.strip() 过滤空字符串
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _load_html() -> str:
    """返回根页面的完整 HTML。

    这里直接返回三引号字符串。
    Python 的三引号字符串允许跨多行书写，适合这种内嵌 HTML 模板。
    """

    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Company Knowledge RAG</title>
  <style>
    :root {
      --bg: #f7f3ea;
      --panel: #fffdfa;
      --ink: #1f1c16;
      --muted: #6b6254;
      --accent: #0e7a6d;
      --accent-2: #b85c38;
      --line: #ddd2c1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(14,122,109,.18), transparent 28%),
        radial-gradient(circle at top right, rgba(184,92,56,.14), transparent 24%),
        var(--bg);
    }
    .shell {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 12px 24px rgba(31,28,22,.05);
    }
    h1, h2 { margin: 0 0 12px 0; }
    h1 { font-size: 28px; }
    h2 { font-size: 18px; }
    p { color: var(--muted); }
    label { display: block; font-size: 14px; margin: 10px 0 6px; }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
    }
    textarea { min-height: 140px; resize: vertical; }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      cursor: pointer;
      margin-top: 12px;
      transition: opacity .2s ease, transform .2s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled {
      cursor: not-allowed;
      opacity: .65;
      transform: none;
    }
    button.secondary { background: var(--accent-2); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stack { display: grid; gap: 14px; }
    .doc-item, .citation, .chunk {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin-top: 10px;
      background: #fff;
    }
    .meta { font-size: 13px; color: var(--muted); }
    .answer {
      white-space: pre-wrap;
      line-height: 1.7;
      font-size: 15px;
      margin-bottom: 16px;
    }
    .status {
      min-height: 22px;
      font-size: 14px;
      color: var(--muted);
    }
    .status.error {
      color: #a33b20;
    }
    .status.success {
      color: var(--accent);
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel stack">
      <div>
        <h1>内部知识库 RAG</h1>
        <p>上传文档、问答、引用、重建索引和检索追踪都在这里。</p>
      </div>
      <div>
        <h2>上传文档</h2>
        <form id="upload-form" class="stack">
          <input type="file" name="file" required />
          <div class="row">
            <input name="owner" placeholder="owner" value="alice" />
            <select name="visibility">
              <option value="private">private</option>
              <option value="shared">shared</option>
            </select>
          </div>
          <div class="row">
            <input name="source" placeholder="source" value="internal" />
            <input name="tags" placeholder="tags,comma,separated" />
          </div>
          <label><input type="checkbox" name="reindex_if_exists" style="width:auto;" /> 同内容强制重建</label>
          <button id="upload-button" type="submit">上传并建索引</button>
          <div id="upload-status" class="status"></div>
        </form>
      </div>
      <div>
        <h2>文档列表</h2>
        <button id="refresh-documents-button" class="secondary" onclick="loadDocuments()">刷新文档</button>
        <div id="documents"></div>
      </div>
    </section>

    <section class="panel stack">
      <div>
        <h2>问答</h2>
        <div class="stack">
          <textarea id="question" placeholder="输入问题，例如：公司知识库里是怎么处理 chunk 大小的？"></textarea>
          <div class="row">
            <input id="user_id" value="alice" />
            <select id="mode">
              <option value="hybrid">hybrid</option>
              <option value="vector">vector</option>
              <option value="bm25">bm25</option>
              <option value="fulltext">fulltext</option>
            </select>
          </div>
          <div class="row">
            <input id="top_k" type="number" min="1" max="20" value="5" />
            <input id="max_context_chunks" type="number" min="1" max="10" value="5" />
          </div>
          <label><input id="enable_rerank" type="checkbox" checked style="width:auto;" /> 启用 rerank</label>
          <button id="ask-button" type="button" onclick="ask()">开始问答</button>
          <div id="ask-status" class="status"></div>
        </div>
      </div>
      <div>
        <h2>回答</h2>
        <div id="answer" class="answer"></div>
        <div id="citations"></div>
      </div>
      <div>
        <h2>命中 Chunk</h2>
        <div id="trace"></div>
      </div>
    </section>
  </div>
  <script>
    // 这三个布尔变量是“前端互斥锁”。
    // 它们的作用是：请求还没回来时，阻止重复点击再次触发同一个动作。
    let askInFlight = false;
    let uploadInFlight = false;
    let documentListBusy = false;

    function setStatus(elementId, message, type = "") {
      // document.getElementById(...) 会根据 id 找页面元素。
      const element = document.getElementById(elementId);
      if (!element) {
        return;
      }
      element.textContent = message || "";
      // 这里用模板字符串拼 className。
      // `status ${type}` 是 JavaScript 模板字符串语法，反引号里可以插 ${变量}。
      element.className = type ? `status ${type}` : "status";
    }

    function setButtonBusy(buttonId, busyText, busy) {
      const button = document.getElementById(buttonId);
      if (!button) {
        return;
      }
      // dataset.defaultLabel 会把原按钮文案缓存起来，
      // 这样忙碌结束后能恢复成最初文字。
      if (!button.dataset.defaultLabel) {
        button.dataset.defaultLabel = button.textContent;
      }
      button.disabled = busy;
      button.textContent = busy ? busyText : button.dataset.defaultLabel;
    }

    function setActionButtonsDisabled(busy) {
      // querySelectorAll 会返回一个 NodeList。
      // forEach(...) 会逐个遍历这些按钮并批量设置 disabled。
      document.querySelectorAll("[data-doc-action='true']").forEach((button) => {
        button.disabled = busy;
      });
    }

    async function loadDocuments() {
      // async function 表示函数内部可以使用 await。
      if (documentListBusy) {
        return;
      }
      documentListBusy = true;
      setButtonBusy("refresh-documents-button", "刷新中...", true);
      try {
        const owner = document.getElementById("user_id")?.value || "alice";
        // ?. 是可选链语法：如果前面取不到元素，不会直接报错，而是返回 undefined。
        const response = await fetch(`/documents?user_id=${encodeURIComponent(owner)}`);
        if (!response.ok) {
          throw new Error(`加载文档失败: ${response.status}`);
        }
        const documents = await response.json();
        const root = document.getElementById("documents");
        root.innerHTML = documents.map(doc => `
          <div class="doc-item">
            <strong>${doc.filename}</strong>
            <div class="meta">owner=${doc.owner} | ${doc.visibility} | version=${doc.version} | chunks=${doc.chunk_count}</div>
            <div class="meta">${doc.updated_at}</div>
            <div class="row">
              <button data-doc-action="true" onclick="reindexDocument('${doc.document_id}', this)">重建索引</button>
              <button data-doc-action="true" class="secondary" onclick="deleteDocument('${doc.document_id}', this)">删除文档</button>
            </div>
          </div>
        `).join("");
        // documents.map(...).join("") 的含义是：
        // 1. map(...) 把每个文档对象变成一段 HTML 字符串
        // 2. join("") 把这些字符串拼成一整个 HTML 片段
      } catch (error) {
        const root = document.getElementById("documents");
        root.innerHTML = `<div class="doc-item">加载失败：${error.message}</div>`;
      } finally {
        // finally 表示“不管成功还是失败都会执行”，常用于收尾逻辑。
        documentListBusy = false;
        setButtonBusy("refresh-documents-button", "刷新中...", false);
      }
    }

    async function uploadDocument(event) {
      // preventDefault() 阻止表单默认刷新页面提交，
      // 改由我们手动发 fetch 请求。
      event.preventDefault();
      if (uploadInFlight) {
        return;
      }
      uploadInFlight = true;
      setButtonBusy("upload-button", "上传中...", true);
      setStatus("upload-status", "正在上传并建立索引，请稍等...", "");
      try {
        const form = document.getElementById("upload-form");
        const formData = new FormData(form);
        formData.set("reindex_if_exists", form.reindex_if_exists.checked ? "true" : "false");
        const response = await fetch("/documents", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || `上传失败: ${response.status}`);
        }
        setStatus(
          "upload-status",
          `上传完成：${result.filename}，生成 ${result.chunk_count} 个 chunk。`,
          "success",
        );
        form.reset();
        await loadDocuments();
      } catch (error) {
        setStatus("upload-status", error.message, "error");
      } finally {
        uploadInFlight = false;
        setButtonBusy("upload-button", "上传中...", false);
      }
    }

    async function ask() {
      if (askInFlight) {
        return;
      }
      askInFlight = true;
      setButtonBusy("ask-button", "问答中...", true);
      setStatus("ask-status", "正在检索并生成回答，请稍等...", "");
      document.getElementById("answer").textContent = "正在处理中...";
      document.getElementById("citations").innerHTML = "";
      document.getElementById("trace").innerHTML = "";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: document.getElementById("question").value,
            user_id: document.getElementById("user_id").value,
            mode: document.getElementById("mode").value,
            top_k: Number(document.getElementById("top_k").value),
            max_context_chunks: Number(document.getElementById("max_context_chunks").value),
            enable_rerank: document.getElementById("enable_rerank").checked
          })
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || `问答失败: ${response.status}`);
        }
        document.getElementById("answer").textContent = result.answer;
        document.getElementById("citations").innerHTML = result.citations.map(item => `
          <div class="citation">
            <strong>${item.filename}</strong>
            <div class="meta">chunk=${item.chunk_index} | page=${item.page ?? "-"} | heading=${item.heading ?? "-"}</div>
            <div>${item.quote}</div>
          </div>
        `).join("");
        document.getElementById("trace").innerHTML = result.retrieval_trace.retrieved_chunks.map(item => `
          <div class="chunk">
            <div class="meta">score=${item.score.toFixed(4)} | file=${item.metadata.filename}</div>
            <div>${item.text}</div>
          </div>
        `).join("");
        setStatus("ask-status", `问答完成，trace_id: ${result.retrieval_trace_id}`, "success");
      } catch (error) {
        document.getElementById("answer").textContent = "";
        setStatus("ask-status", error.message, "error");
      } finally {
        askInFlight = false;
        setButtonBusy("ask-button", "问答中...", false);
      }
    }

    async function deleteDocument(documentId, button) {
      if (!button || button.disabled) {
        return;
      }
      button.disabled = true;
      const original = button.textContent;
      button.textContent = "删除中...";
      setActionButtonsDisabled(true);
      try {
        const response = await fetch(`/documents/${documentId}`, { method: "DELETE" });
        if (!response.ok) {
          const result = await response.json();
          throw new Error(result.detail || `删除失败: ${response.status}`);
        }
        await loadDocuments();
      } catch (error) {
        alert(error.message);
      } finally {
        setActionButtonsDisabled(false);
        button.disabled = false;
        button.textContent = original;
      }
    }

    async function reindexDocument(documentId, button) {
      if (!button || button.disabled) {
        return;
      }
      button.disabled = true;
      const original = button.textContent;
      button.textContent = "重建中...";
      setActionButtonsDisabled(true);
      try {
        const response = await fetch(`/documents/${documentId}/reindex`, { method: "POST" });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || `重建失败: ${response.status}`);
        }
        alert(JSON.stringify(result, null, 2));
        await loadDocuments();
      } catch (error) {
        alert(error.message);
      } finally {
        setActionButtonsDisabled(false);
        button.disabled = false;
        button.textContent = original;
      }
    }

    document.getElementById("upload-form").addEventListener("submit", uploadDocument);
    loadDocuments();
  </script>
</body>
</html>
"""
