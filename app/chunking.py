from __future__ import annotations

from dataclasses import dataclass
import re

from app.parsers import ParsedSection


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    """准备写入向量库的切块结果。"""

    text: str
    chunk_index: int
    page: int | None
    heading: str | None
    token_count: int


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数；教学项目里先不用 tokenizer，避免模型依赖。"""

    return len(TOKEN_RE.findall(text))


def chunk_sections(
    sections: list[ParsedSection],
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """把解析后的文档切成适合检索的 chunk。

    策略：先按段落和 Markdown 标题聚合，超过上限再按 token 窗口切分。
    这样比固定字符切分更容易保留标题、段落和代码块的上下文。
    """

    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("CHUNK_OVERLAP_TOKENS 必须小于 CHUNK_SIZE_TOKENS。")

    chunks: list[Chunk] = []
    active_heading: str | None = None

    for section in sections:
        blocks = _split_blocks(section.text)
        buffer: list[str] = []
        buffer_tokens = 0
        buffer_heading: str | None = None

        for block in blocks:
            heading = _extract_heading(block)
            if heading and buffer:
                # 新标题通常代表语义边界，先结束上一块，避免 metadata 被新标题污染。
                chunks.extend(
                    _flush_buffer(buffer, len(chunks), section.page, buffer_heading)
                )
                buffer = []
                buffer_tokens = 0
                buffer_heading = None

            if heading:
                active_heading = heading

            block_heading = heading or active_heading

            block_tokens = estimate_tokens(block)
            if block_tokens > chunk_size_tokens:
                chunks.extend(
                    _flush_buffer(buffer, len(chunks), section.page, buffer_heading)
                )
                buffer = []
                buffer_tokens = 0
                buffer_heading = None
                chunks.extend(
                    _split_large_block(
                        block,
                        len(chunks),
                        section.page,
                        block_heading,
                        chunk_size_tokens,
                        overlap_tokens,
                    )
                )
                continue

            if buffer and buffer_tokens + block_tokens > chunk_size_tokens:
                chunks.extend(
                    _flush_buffer(buffer, len(chunks), section.page, buffer_heading)
                )
                # 保留上一块末尾一点内容，减少答案线索刚好落在边界时的召回损失。
                buffer = _tail_overlap(buffer, overlap_tokens)
                buffer_tokens = estimate_tokens("\n\n".join(buffer))
                buffer_heading = buffer_heading if buffer else block_heading

            if not buffer:
                buffer_heading = block_heading
            buffer.append(block)
            buffer_tokens += block_tokens

        chunks.extend(_flush_buffer(buffer, len(chunks), section.page, buffer_heading))

    return chunks


def _split_blocks(text: str) -> list[str]:
    """按空行切段，同时尽量保持代码块不被段落切分破坏。"""

    blocks: list[str] = []
    current: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        if not in_code_block and not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _extract_heading(block: str) -> str | None:
    first_line = block.splitlines()[0].strip()
    match = HEADING_RE.match(first_line)
    return match.group(2).strip() if match else None


def _flush_buffer(
    buffer: list[str],
    start_index: int,
    page: int | None,
    heading: str | None,
) -> list[Chunk]:
    if not buffer:
        return []
    text = "\n\n".join(buffer).strip()
    return [
        Chunk(
            text=text,
            chunk_index=start_index,
            page=page,
            heading=heading,
            token_count=estimate_tokens(text),
        )
    ]


def _split_large_block(
    block: str,
    start_index: int,
    page: int | None,
    heading: str | None,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    tokens = TOKEN_RE.findall(block)
    chunks: list[Chunk] = []
    step = chunk_size_tokens - overlap_tokens

    # 大段内容退化为 token 窗口切分，保证任何单段都不会无限大。
    for offset in range(0, len(tokens), step):
        window = tokens[offset : offset + chunk_size_tokens]
        if not window:
            continue
        text = " ".join(window)
        chunks.append(
            Chunk(
                text=text,
                chunk_index=start_index + len(chunks),
                page=page,
                heading=heading,
                token_count=len(window),
            )
        )
    return chunks


def _tail_overlap(buffer: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    selected: list[str] = []
    total = 0
    for block in reversed(buffer):
        selected.append(block)
        total += estimate_tokens(block)
        if total >= overlap_tokens:
            break
    return list(reversed(selected))
