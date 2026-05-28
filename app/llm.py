from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error, request

from app.config import Settings
from app.text_utils import keyword_query


NO_ANSWER_TOKEN = "INSUFFICIENT_CONTEXT"
LOGGER = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """通过 OpenAI 兼容接口完成查询改写与答案生成。"""

    def __init__(self, settings: Settings):
        self.base_url = settings.openai_base_url.rstrip("/") if settings.openai_base_url else None
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.user_agent = settings.openai_user_agent
        self.timeout_seconds = settings.openai_timeout_seconds
        # last_error 会保留最近一次失败原因，供 /health 和排查使用。
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        # bool(a and b and c) 利用了 Python 的布尔短路规则：
        # 只要其中一个为空，就会得到假值；全部存在时才返回 True。
        return bool(self.base_url and self.api_key and self.model)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.enabled:
            self.last_error = "LLM 未启用：缺少 base_url、api_key 或 model。"
            return None

        # 这里故意不用 response_format=json_object。
        # 原因是很多兼容网关对这个参数支持不稳定，
        # 我们改成“普通请求 + 自己解析 content 里的 JSON”，兼容性更高。
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post_json(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
        )
        if data is None:
            return None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            self.last_error = f"LLM 响应缺少 choices/message/content: {exc}"
            LOGGER.warning(self.last_error)
            return None

        if isinstance(content, list):
            # 一些兼容接口会把 content 返回成多个片段。
            # "".join(...) 会把这些字符串片段拼成一个完整文本。
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        if not isinstance(content, str):
            self.last_error = "LLM content 不是字符串，无法解析。"
            LOGGER.warning(self.last_error)
            return None

        parsed = self._extract_json_object(content)
        if parsed is None:
            self.last_error = f"LLM content 不是可解析 JSON: {content[:200]}"
            LOGGER.warning(self.last_error)
            return None

        self.last_error = None
        return parsed

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """发送兼容请求时显式带上 Accept 和 User-Agent，兼容中转网关。"""

        # ensure_ascii=False 可以保留中文提示词，
        # 否则 JSON 会把中文转成 \uXXXX 的形式，可读性会变差。
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = ""
            try:
                # exc.read() 会继续读取服务端返回的错误体，
                # 对排查模型名错误、鉴权错误、参数错误很有帮助。
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            self.last_error = f"LLM HTTPError {exc.code}: {detail[:300]}"
            LOGGER.warning(self.last_error)
            return None
        except error.URLError as exc:
            self.last_error = f"LLM URLError: {exc.reason}"
            LOGGER.warning(self.last_error)
            return None
        except TimeoutError:
            self.last_error = "LLM 调用超时。"
            LOGGER.warning(self.last_error)
            return None
        except json.JSONDecodeError as exc:
            self.last_error = f"LLM 原始响应不是合法 JSON: {exc}"
            LOGGER.warning(self.last_error)
            return None

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any] | None:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 有些兼容模型会返回“解释文字 + JSON”的混合文本，
        # 所以这里退一步，只取最外层大括号之间的内容再试一次。
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None


class QueryPlanner:
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        enable_query_rewrite: bool,
        enable_multi_query: bool,
    ):
        self.llm_client = llm_client
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_multi_query = enable_multi_query

    def build_queries(self, question: str) -> list[str]:
        keyword_version = keyword_query(question)
        queries = [question]
        if self.enable_query_rewrite:
            rewritten = self._rewrite_query(question)
            if rewritten and rewritten not in queries:
                queries.append(rewritten)
        if self.enable_multi_query and keyword_version and keyword_version not in queries:
            queries.append(keyword_version)

        # queries[:3] 是切片语法，表示最多保留前三个检索查询。
        return queries[:3]

    def _rewrite_query(self, question: str) -> str | None:
        result = self.llm_client.chat_json(
            system_prompt=(
                "你负责把用户问题改写成更适合检索的查询。"
                "输出 JSON：{\"rewritten_query\": \"...\"}。"
            ),
            user_prompt=question,
        )
        if result and isinstance(result.get("rewritten_query"), str):
            return result["rewritten_query"].strip()
        return keyword_query(question)


class AnswerGenerator:
    def __init__(self, llm_client: OpenAICompatibleClient, enable_llm_answer: bool):
        self.llm_client = llm_client
        self.enable_llm_answer = enable_llm_answer

    def generate(
        self,
        *,
        question: str,
        contexts: list[dict[str, Any]],
        force_no_answer: bool,
    ) -> tuple[str, bool]:
        # tuple[str, bool] 表示同时返回“答案文本”和“是否 no_answer”。
        if force_no_answer or not contexts:
            return "查不到足够可靠的内容来回答这个问题。", True

        if not self.enable_llm_answer or not self.llm_client.enabled:
            return self._fallback_answer(contexts), False

        context_lines = []
        for index, item in enumerate(contexts, start=1):
            # enumerate(..., start=1) 会在遍历时同时给出序号，
            # 这里从 1 开始，是为了让引用编号更接近人类阅读习惯。
            metadata = item["metadata"]
            context_lines.append(
                f"[{index}] 文件: {metadata.get('filename')} | 页码: {metadata.get('page')} | 标题: {metadata.get('heading')}\n"
                f"{item['text']}"
            )

        result = self.llm_client.chat_json(
            system_prompt=(
                "你是公司内部知识库问答助手。"
                f"如果上下文不足，必须返回 JSON: {{\"answer\": \"{NO_ANSWER_TOKEN}\"}}。"
                "如果可以回答，输出 JSON: {\"answer\": \"简洁中文答案\"}。"
                "只能依据提供的上下文回答，不要编造。"
            ),
            user_prompt=f"问题:\n{question}\n\n上下文:\n" + "\n\n".join(context_lines),
        )
        if result and isinstance(result.get("answer"), str):
            answer = result["answer"].strip()
            if answer == NO_ANSWER_TOKEN:
                return "查不到足够可靠的内容来回答这个问题。", True
            return answer, False
        return self._fallback_answer(contexts), False

    @staticmethod
    def _fallback_answer(contexts: list[dict[str, Any]]) -> str:
        # fallback 不是“真正推理”，只是把最相关片段整理成一个可读摘要。
        excerpts = [item["text"][:180].strip() for item in contexts[:2]]
        joined = "\n".join(f"- {excerpt}" for excerpt in excerpts if excerpt)
        return f"基于检索到的内容，当前最相关的信息如下：\n{joined}"
