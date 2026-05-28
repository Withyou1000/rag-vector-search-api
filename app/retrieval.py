from __future__ import annotations

from collections import Counter
import math
from typing import Any

from app.config import Settings
from app.llm import AnswerGenerator, QueryPlanner
from app.storage import DocumentRepository, TraceRepository
from app.text_utils import overlap_score, tokenize
from app.vector_store import VectorStore


class RetrievalService:
    """统一编排 dense、fulltext、BM25、hybrid 与 rerank。"""

    def __init__(
        self,
        *,
        settings: Settings,
        vector_store: VectorStore,
        document_repository: DocumentRepository,
        trace_repository: TraceRepository,
        query_planner: QueryPlanner,
        answer_generator: AnswerGenerator,
    ):
        # 这些依赖都由 app 启动时一次性注入，后面 search/ask 直接复用。
        self.settings = settings
        self.vector_store = vector_store
        self.document_repository = document_repository
        self.trace_repository = trace_repository
        self.query_planner = query_planner
        self.answer_generator = answer_generator

    def search(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
        enable_rerank: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # build_queries 可能会返回“原问题 + 改写问题 + 关键词问题”。
        # 这样做的目的是让多路检索在口语问题上也有更高召回率。
        used_queries = self.query_planner.build_queries(query)

        # candidate_limit 用 max(...) 取较大值：
        # 1. top_k 是前端最终想看的条数
        # 2. default_candidate_limit 是内部排序想保留的候选池大小
        # 两者取大，是为了先多召回一些结果，再裁剪成最终输出。
        candidate_limit = max(top_k, self.settings.default_candidate_limit)

        if mode == "vector":
            # vector 模式只走向量检索，用第一条查询就够了。
            hits = self._vector_candidates(
                used_queries[0],
                candidate_limit,
                source,
                tags,
                user_id,
            )
            fused = hits
        elif mode == "fulltext":
            # fulltext 更像“包含这些词没有”，适合明确关键词或短语。
            fused = self._fulltext_search(used_queries[0], source, tags, user_id)
        elif mode == "bm25":
            # bm25 更关注词频、稀有度和文档长度平衡，适合关键词排序。
            fused = self._bm25_search(used_queries[0], source, tags, user_id)
        elif mode == "hybrid":
            # hybrid 先保留 dense 召回，抓“语义上相关”的候选。
            vector_hits = self._vector_candidates(
                used_queries[0],
                candidate_limit,
                source,
                tags,
                user_id,
            )

            # lexical_hits 会汇总 BM25 与 fulltext 两路结果。
            lexical_hits: list[dict[str, Any]] = []
            for retrieval_query in used_queries:
                # 这里故意让每个 retrieval_query 都参与 lexical 检索，
                # 因为改写问题和关键词问题往往比原问题更适合命中文档里的显式文本。
                lexical_hits.extend(
                    self._bm25_search(retrieval_query, source, tags, user_id)
                )
                lexical_hits.extend(
                    self._fulltext_search(retrieval_query, source, tags, user_id)
                )

            # RRF（Reciprocal Rank Fusion）不直接比不同检索分数的绝对值，
            # 而是看每一路里“排第几名”，特别适合融合 dense 和 lexical。
            fused = self._rrf_fuse([vector_hits, lexical_hits])
        else:
            raise ValueError(f"不支持的检索模式: {mode}")

        # rerank 是一个“候选重排”步骤：
        # 如果启用，就在已有候选上再加一层 query-text 的词项重叠分。
        reranked = self._rerank(query=query, hits=fused) if enable_rerank else fused

        # 最终只把前 top_k 条返回给前端。
        final_hits = reranked[:top_k]

        # trace_payload 会被 /search 返回一部分，也会被 /ask 存成检索追踪。
        trace_payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "user_id": user_id,
            "used_queries": used_queries,
            "enable_rerank": enable_rerank,
            "retrieved_chunks": [self._trace_hit(hit) for hit in final_hits],
        }
        return final_hits, trace_payload

    def ask(
        self,
        *,
        question: str,
        mode: str,
        top_k: int,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
        enable_rerank: bool,
        max_context_chunks: int,
    ) -> dict[str, Any]:
        # ask 先调用 search，保证“单独搜索”和“问答前搜索”走的是同一套逻辑。
        hits, trace_payload = self.search(
            query=question,
            mode=mode,
            top_k=top_k,
            source=source,
            tags=tags,
            user_id=user_id,
            enable_rerank=enable_rerank,
        )

        # _compress_context 不是再次检索，而是把已命中的 chunk 压成更适合给 LLM 的上下文。
        contexts = self._compress_context(hits, max_context_chunks)

        # no-answer 判断单独拆到 _is_low_confidence，
        # 这样后面调策略时不会和答案生成逻辑缠在一起。
        low_confidence = self._is_low_confidence(
            question=question,
            mode=mode,
            hits=hits,
        )

        # generate 返回 tuple[str, bool]：
        # 第一个值是答案文本，第二个值是 no_answer 标记。
        answer, no_answer = self.answer_generator.generate(
            question=question,
            contexts=contexts,
            force_no_answer=low_confidence,
        )

        trace_payload["final_context_chunks"] = [self._trace_hit(item) for item in contexts]
        trace_payload["answer"] = answer
        trace_payload["no_answer"] = no_answer
        trace_id = self.trace_repository.save(trace_payload)
        citations = [self._citation_from_hit(item) for item in contexts]
        return {
            "answer": answer,
            "citations": citations,
            "retrieval_trace_id": trace_id,
            "used_queries": trace_payload["used_queries"],
            "no_answer": no_answer,
            "retrieval_trace": trace_payload,
        }

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.trace_repository.get(trace_id)

    def _vector_candidates(
        self,
        query: str,
        top_k: int,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return self.vector_store.search(
            query=query,
            top_k=top_k,
            source=source,
            tags=tags,
            user_id=user_id,
        )

    def _fulltext_search(
        self,
        query: str,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        chunks = self.vector_store.list_chunks(source=source, tags=tags, user_id=user_id)
        normalized_tokens = tokenize(query)
        hits: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str(chunk["text"]).lower()

            # sum(1 for token in normalized_tokens if token in text) 是生成器表达式：
            # 1. for token in normalized_tokens：逐个遍历查询词
            # 2. if token in text：只保留当前 chunk 里出现的词
            # 3. sum(...)：把每个命中的 1 累加起来，得到命中词数
            term_matches = sum(1 for token in normalized_tokens if token in text)
            if term_matches == 0:
                continue

            # term_matches / 查询词数 是一个朴素覆盖率分数。
            score = term_matches / max(len(normalized_tokens), 1)
            hits.append(self._chunk_to_hit(chunk=chunk, score=score))
        return sorted(hits, key=lambda item: item["score"], reverse=True)

    def _bm25_search(
        self,
        query: str,
        source: str | None,
        tags: list[str] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        chunks = self.vector_store.list_chunks(source=source, tags=tags, user_id=user_id)
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        corpus_tokens = [tokenize(str(chunk["text"])) for chunk in chunks]
        if not corpus_tokens:
            return []

        avg_len = sum(len(tokens) for tokens in corpus_tokens) / len(corpus_tokens)
        doc_freq = Counter()
        for tokens in corpus_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        hits: list[dict[str, Any]] = []
        for chunk, tokens in zip(chunks, corpus_tokens):
            # zip(chunks, corpus_tokens) 会把“原始 chunk”和“对应分词结果”按位置配对。
            token_counts = Counter(tokens)
            doc_len = max(len(tokens), 1)
            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                df = doc_freq[token]

                # idf 是 inverse document frequency，表示这个词在全库里有多稀有。
                idf = math.log((len(corpus_tokens) - df + 0.5) / (df + 0.5) + 1)
                tf = token_counts[token]

                # 下面两行是 BM25 公式拆解：
                # numerator 体现词频贡献；
                # denominator 负责做长度归一化，避免长文档天然吃分。
                numerator = tf * 2.2
                denominator = tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1))
                score += idf * (numerator / denominator)
            if score > 0:
                hits.append(self._chunk_to_hit(chunk=chunk, score=score))
        return sorted(hits, key=lambda item: item["score"], reverse=True)

    def _rrf_fuse(
        self,
        ranked_lists: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        fused_scores: dict[str, float] = {}
        source_hit: dict[str, dict[str, Any]] = {}
        for ranked in ranked_lists:
            for index, hit in enumerate(ranked, start=1):
                chunk_id = str(hit["metadata"]["chunk_id"])

                # 1 / (k + index) 是 RRF 的核心思想：
                # 排名越靠前，贡献越大；但因为有 k 做平滑，不会让第一名“一票否决”。
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (k + index)
                source_hit[chunk_id] = hit

        results = []
        for chunk_id, score in sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            hit = dict(source_hit[chunk_id])
            hit["score"] = score
            results.append(hit)
        return results

    def _rerank(self, *, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranked = []
        for hit in hits:
            # overlap_score 不是替代原始分数，而是额外加一个 lexical boost。
            lexical_boost = overlap_score(query, hit["text"])
            reranked.append({**hit, "score": hit["score"] + lexical_boost})
        return sorted(reranked, key=lambda item: item["score"], reverse=True)

    def _compress_context(
        self,
        hits: list[dict[str, Any]],
        max_context_chunks: int,
    ) -> list[dict[str, Any]]:
        contexts = []
        seen = set()
        for hit in hits:
            chunk_id = hit["metadata"]["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            text = hit["text"][: self.settings.max_context_chars_per_chunk].strip()
            contexts.append({**hit, "text": text})
            if len(contexts) >= max_context_chunks:
                break
        return contexts

    def _is_low_confidence(
        self,
        *,
        question: str,
        mode: str,
        hits: list[dict[str, Any]],
    ) -> bool:
        """判断是否应在生成前直接返回 no-answer。"""

        if not hits or len(hits) < self.settings.min_answer_hits:
            return True

        # 显式转成 float，是为了避免底层返回 numpy 数值类型时比较行为不直观。
        top_score = float(hits[0]["score"])

        # evidence_score 不是新的检索分数，而是“辅助证据”。
        # 这里只看前 3 条里 query-text 重叠度最高的一条，
        # 目的是判断“前排候选里是否至少有一个明显贴题”。
        evidence_score = max(
            overlap_score(question, hit["text"])
            for hit in hits[: min(3, len(hits))]
        )

        # 这里不是只看配置文件有没有 key，而是看：
        # 1. enable_llm_answer 没被关闭
        # 2. llm_client.enabled 真的为真
        llm_is_available = (
            self.answer_generator.enable_llm_answer
            and self.answer_generator.llm_client.enabled
        )

        if mode == "vector":
            # vector 模式仍然保留阈值保护，
            # 避免语义相似但其实答非所问的 chunk 直接放进生成阶段。
            return (
                top_score < self.settings.answer_score_threshold
                and evidence_score < 0.12
            )

        if llm_is_available:
            # 对 hybrid/fulltext/bm25 来说，分数口径和 vector 不同，
            # 尤其 hybrid 的 RRF 分数天然偏小，所以这里不再硬套统一阈值。
            # 只要 LLM 可用，就让模型继续基于上下文判断是否能回答。
            return False

        # 没有 LLM 时只能依赖检索证据本身，
        # 所以保留一个轻量保护，避免 fallback 在不相关命中上硬拼摘要。
        return evidence_score < 0.12

    @staticmethod
    def _trace_hit(hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": hit["score"],
            "text": hit["text"],
            "metadata": hit["metadata"],
        }

    @staticmethod
    def _citation_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
        metadata = hit["metadata"]
        return {
            "document_id": metadata.get("document_id"),
            "filename": metadata.get("filename"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_index": metadata.get("chunk_index"),
            "page": metadata.get("page"),
            "heading": metadata.get("heading"),
            "quote": hit["text"][:220].strip(),
        }

    @staticmethod
    def _chunk_to_hit(chunk: dict[str, Any], score: float) -> dict[str, Any]:
        # 这里用字典推导式过滤掉 text，
        # 因为 text 会作为一级字段单独返回，metadata 里不再重复塞一遍。
        metadata = {key: value for key, value in chunk.items() if key != "text"}
        return {"score": score, "text": chunk["text"], "metadata": metadata}
