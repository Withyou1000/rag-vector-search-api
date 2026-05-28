from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    source: str | None
    tags: list[str]
    owner: str
    visibility: str
    version: int
    status: str
    updated_at: str
    content_hash: str
    chunk_count: int
    source_path: str


class DocumentRepository:
    """用本地 JSON 保存文档元数据，让版本、权限和重建索引更可控。"""

    def __init__(self, metadata_path: Path, documents_dir: Path):
        self.metadata_path = metadata_path
        self.documents_dir = documents_dir
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            self._write({"documents": []})

    def list_documents(self, owner: str | None = None) -> list[StoredDocument]:
        documents = [self._from_dict(item) for item in self._read()["documents"]]
        active_documents = [item for item in documents if item.status == "active"]
        if owner is None:
            return active_documents
        return [item for item in active_documents if item.owner == owner or item.visibility == "shared"]

    def get_document(self, document_id: str) -> StoredDocument | None:
        for item in self._read()["documents"]:
            if item["document_id"] == document_id and item["status"] != "deleted":
                return self._from_dict(item)
        return None

    def get_by_owner_and_filename(self, owner: str, filename: str) -> StoredDocument | None:
        candidates = [
            self._from_dict(item)
            for item in self._read()["documents"]
            if item["owner"] == owner and item["filename"] == filename and item["status"] == "active"
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.version, reverse=True)[0]

    def save_upload(
        self,
        *,
        filename: str,
        content: bytes,
        source: str | None,
        tags: list[str],
        owner: str,
        visibility: str,
        chunk_count: int,
        reindex_if_exists: bool,
    ) -> tuple[StoredDocument, bool]:
        content_hash = sha256(content).hexdigest()
        data = self._read()
        existing = self.get_by_owner_and_filename(owner=owner, filename=filename)
        if existing and existing.content_hash == content_hash and not reindex_if_exists:
            existing.tags = tags or existing.tags
            existing.source = source or existing.source
            self._upsert_document(data, existing)
            self._write(data)
            return existing, False

        if existing:
            document_id = existing.document_id
            version = existing.version + 1
        else:
            document_id = str(uuid4())
            version = 1

        source_path = self._persist_source(document_id=document_id, version=version, filename=filename, content=content)
        document = StoredDocument(
            document_id=document_id,
            filename=filename,
            source=source,
            tags=tags,
            owner=owner,
            visibility=visibility,
            version=version,
            status="active",
            updated_at=utc_now_iso(),
            content_hash=content_hash,
            chunk_count=chunk_count,
            source_path=source_path,
        )
        self._upsert_document(data, document)
        self._write(data)
        return document, True

    def update_chunk_count(self, document_id: str, chunk_count: int) -> StoredDocument:
        data = self._read()
        document = self.get_document(document_id)
        if document is None:
            raise ValueError(f"文档不存在: {document_id}")
        document.chunk_count = chunk_count
        document.updated_at = utc_now_iso()
        self._upsert_document(data, document)
        self._write(data)
        return document

    def mark_deleted(self, document_id: str) -> StoredDocument:
        data = self._read()
        document = self.get_document(document_id)
        if document is None:
            raise ValueError(f"文档不存在: {document_id}")
        document.status = "deleted"
        document.updated_at = utc_now_iso()
        self._upsert_document(data, document)
        self._write(data)
        return document

    def bump_version_for_reindex(self, document_id: str) -> StoredDocument:
        data = self._read()
        document = self.get_document(document_id)
        if document is None:
            raise ValueError(f"文档不存在: {document_id}")
        source_bytes = Path(document.source_path).read_bytes()
        next_version = document.version + 1
        source_path = self._persist_source(
            document_id=document.document_id,
            version=next_version,
            filename=document.filename,
            content=source_bytes,
        )
        document.version = next_version
        document.source_path = source_path
        document.updated_at = utc_now_iso()
        self._upsert_document(data, document)
        self._write(data)
        return document

    def _persist_source(self, *, document_id: str, version: int, filename: str, content: bytes) -> str:
        target_dir = self.documents_dir / document_id / f"v{version}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_bytes(content)
        return str(target_path)

    def _upsert_document(self, data: dict[str, Any], document: StoredDocument) -> None:
        documents = data["documents"]
        for index, item in enumerate(documents):
            if item["document_id"] == document.document_id:
                documents[index] = asdict(document)
                return
        documents.append(asdict(document))

    def _read(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self.metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> StoredDocument:
        return StoredDocument(**data)


class TraceRepository:
    """按文件保存 retrieval trace，方便评测和人工排查。"""

    def __init__(self, traces_dir: Path):
        self.traces_dir = traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict[str, Any]) -> str:
        trace_id = str(uuid4())
        body = {"trace_id": trace_id, "created_at": utc_now_iso(), **payload}
        (self.traces_dir / f"{trace_id}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return trace_id

    def get(self, trace_id: str) -> dict[str, Any] | None:
        path = self.traces_dir / f"{trace_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
