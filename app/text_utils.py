from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """统一小写与空白，减少检索时的无关差异。"""

    return " ".join(text.lower().split())


def tokenize(text: str) -> list[str]:
    """统一 dense 与 lexical 相关组件的分词口径。"""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def keyword_query(text: str, limit: int = 8) -> str:
    """把口语问题收缩成更适合 lexical 检索的关键词串。"""

    counts = Counter(tokenize(text))
    keywords = [token for token, _ in counts.most_common(limit)]
    return " ".join(keywords)


def overlap_score(query: str, text: str) -> float:
    """没有真实 reranker 时，用词项重叠做一个轻量排序信号。"""

    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / math.sqrt(len(query_tokens) * len(text_tokens))
