from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class ParsedSection:
    """解析后的基础文本单元，PDF 会保留页码，文本文件默认只有第 1 页。"""

    text: str
    page: int | None = None


def parse_document(filename: str, content: bytes) -> list[ParsedSection]:
    """按文件类型解析文档，并统一返回可切块的文本片段。"""

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持 {extension}，请上传 Markdown、txt 或 PDF。")

    if extension == ".pdf":
        return _parse_pdf(content)
    return _parse_text(content)


def _parse_text(content: bytes) -> list[ParsedSection]:
    # 先尝试 UTF-8，失败后回退到 gbk，兼容常见中文 Windows 文本文件。
    for encoding in ("utf-8", "gbk"):
        try:
            return [ParsedSection(text=content.decode(encoding), page=1)]
        except UnicodeDecodeError:
            continue
    raise ValueError("文本文件编码无法识别，请转换为 UTF-8 后再上传。")


def _parse_pdf(content: bytes) -> list[ParsedSection]:
    # PyMuPDF 只在解析 PDF 时导入，避免纯文本场景承担额外启动成本。
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("请先安装 PyMuPDF，才能解析 PDF 文件。") from exc

    sections: list[ParsedSection] = []
    with fitz.open(stream=content, filetype="pdf") as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                sections.append(ParsedSection(text=text, page=page_index))

    if not sections:
        raise ValueError("PDF 中没有解析出文本，可能是扫描件或图片型 PDF。")
    return sections
