"""Milvus vector index for versioned, tenant-aware managed SOP chunks.

SQLite remains the source of truth for documents, versions and permissions.
This module owns only the derived vector index and can be rebuilt at any time.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import requests
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from security import ROLE_LEVEL


COLLECTION_NAME = os.environ.get(
    "MANAGED_KNOWLEDGE_COLLECTION", "managed_sop_knowledge"
)
CONNECTION_ALIAS = "managed_sop_conn"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "2560"))


def _safe_expr(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


class ManagedKnowledgeVector:
    """Derived Milvus index used by the managed SOP document library."""

    def __init__(self) -> None:
        self.enabled = os.environ.get(
            "MANAGED_KNOWLEDGE_VECTOR_ENABLED", "true"
        ).lower() == "true"
        self._lock = threading.RLock()
        self._connected = False
        self._last_error = ""

    def _connect(self) -> None:
        if not self.enabled:
            raise RuntimeError("受控 SOP Milvus 向量索引已禁用")
        if self._connected and connections.has_connection(CONNECTION_ALIAS):
            return
        with self._lock:
            if self._connected and connections.has_connection(CONNECTION_ALIAS):
                return
            host = os.environ.get("MILVUS_HOST", "").strip()
            port = os.environ.get("MILVUS_PORT", "19530").strip()
            user = os.environ.get("MILVUS_USER", "root").strip()
            password = os.environ.get("MILVUS_PASSWORD", "")
            uri = os.environ.get("URL", "").strip()
            token = os.environ.get("Token", "")
            if host:
                kwargs: dict[str, Any] = {
                    "alias": CONNECTION_ALIAS, "host": host, "port": port,
                }
                if user:
                    kwargs["user"] = user
                if password:
                    kwargs["password"] = password
                connections.connect(**kwargs)
            elif uri:
                kwargs = {"alias": CONNECTION_ALIAS, "uri": uri}
                if token:
                    kwargs["token"] = token
                connections.connect(**kwargs)
            else:
                local_path = os.path.abspath("output/milvus_local.db")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                connections.connect(alias=CONNECTION_ALIAS, uri=local_path)
            self._connected = True

    def _collection(self) -> Collection:
        self._connect()
        if not utility.has_collection(COLLECTION_NAME, using=CONNECTION_ALIAS):
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True,
                    auto_id=False, max_length=64,
                ),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                ),
                FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="version_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="version", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="min_role", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            ]
            schema = CollectionSchema(
                fields, description="受控 SOP 文档当前有效版本的向量分块"
            )
            collection = Collection(
                name=COLLECTION_NAME, schema=schema,
                using=CONNECTION_ALIAS, consistency_level="Strong",
            )
            collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "IP", "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            return collection
        return Collection(name=COLLECTION_NAME, using=CONNECTION_ALIAS)

    def _embedding_request(self, inputs: list[str]) -> list[list[float]]:
        api_key = os.environ.get(
            "LLM_API_KEY", os.environ.get("ARK_API_KEY", "")
        )
        base_url = os.environ.get(
            "LLM_BASE_URL",
            os.environ.get("ARK_BASE_URL", "https://api.siliconflow.cn/v1"),
        ).rstrip("/")
        if not api_key:
            raise RuntimeError("缺少 LLM_API_KEY，无法生成受控 SOP embedding")
        response = requests.post(
            f"{base_url}/embeddings",
            json={"model": EMBEDDING_MODEL, "input": inputs},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=int(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60")),
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in rows]
        if len(vectors) != len(inputs):
            raise RuntimeError("Embedding 服务返回的向量数量与输入不一致")
        if vectors and len(vectors[0]) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding 维度不匹配：返回 {len(vectors[0])}，配置 {EMBEDDING_DIM}"
            )
        return vectors

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "16")))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embedding_request(texts[start:start + batch_size]))
        return vectors

    def delete_document(self, tenant_id: str, document_id: str) -> int:
        collection = self._collection()
        expr = (
            f'tenant_id == "{_safe_expr(tenant_id)}" && '
            f'document_id == "{_safe_expr(document_id)}"'
        )
        result = collection.delete(expr)
        collection.flush()
        return int(getattr(result, "delete_count", 0) or 0)

    def sync_document(self, payload: dict) -> dict:
        """Replace all vectors for one document with its current active version."""
        chunks = list(payload.get("chunks") or [])
        tenant_id = str(payload["tenant_id"])
        document_id = str(payload["document_id"])
        self.delete_document(tenant_id, document_id)
        if not chunks or payload.get("document_status") != "active":
            return {"backend": "milvus", "collection": COLLECTION_NAME, "chunks": 0}

        title = str(payload.get("title") or "")[:512]
        vectors = self.embed_many([
            f"{title}\n{str(item.get('content') or '')}" for item in chunks
        ])
        collection = self._collection()
        data = [
            [str(item["id"]) for item in chunks],
            vectors,
            [tenant_id] * len(chunks),
            [document_id] * len(chunks),
            [str(payload["version_id"])] * len(chunks),
            [int(payload["version"])] * len(chunks),
            [int(item["chunk_index"]) for item in chunks],
            [str(payload.get("min_role") or "viewer")] * len(chunks),
            [title] * len(chunks),
            [str(item.get("location") or "")[:512] for item in chunks],
            [str(item.get("content") or "")[:8192] for item in chunks],
        ]
        collection.insert(data)
        collection.flush()
        self._last_error = ""
        return {
            "backend": "milvus", "collection": COLLECTION_NAME,
            "chunks": len(chunks), "version_id": payload["version_id"],
        }

    def search(
        self, tenant_id: str, role: str, query: str, limit: int = 5,
        document_ids: set[str] | None = None,
    ) -> list[dict]:
        if not query.strip() or limit <= 0:
            return []
        if document_ids is not None and not document_ids:
            return []
        collection = self._collection()
        collection.load()
        vector = self.embed_many([query])[0]
        allowed_roles = [
            name for name, level in ROLE_LEVEL.items()
            if level <= ROLE_LEVEL.get(role, 0)
        ]
        roles_expr = ",".join(f'"{_safe_expr(item)}"' for item in allowed_roles)
        expr = (
            f'tenant_id == "{_safe_expr(tenant_id)}" && '
            f"min_role in [{roles_expr}]"
        )
        if document_ids is not None:
            ids_expr = ",".join(
                f'"{_safe_expr(item)}"' for item in sorted(document_ids)
            )
            expr += f" && document_id in [{ids_expr}]"
        results = collection.search(
            data=[vector], anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=max(1, min(limit, 20)), expr=expr,
            output_fields=[
                "document_id", "version_id", "version", "chunk_index", "title",
                "location", "content", "min_role",
            ],
        )
        hits = []
        for hit in results[0] if results else []:
            hits.append({
                "evidence_id": f"sop:{hit.id}",
                "source_type": "managed_sop",
                "document_id": hit.entity.get("document_id"),
                "version_id": hit.entity.get("version_id"),
                "version": hit.entity.get("version"),
                "title": hit.entity.get("title"),
                "location": hit.entity.get("location"),
                "content": hit.entity.get("content"),
                "score": round(float(hit.distance), 4),
                "trust_level": "authoritative",
                "retrieval_backend": "milvus",
            })
        self._last_error = ""
        return hits

    def health(self) -> dict:
        return {
            "ok": not bool(self._last_error),
            "enabled": self.enabled,
            "backend": "milvus",
            "collection": COLLECTION_NAME,
            "connected": self._connected,
            "last_error": self._last_error,
        }

    def remember_error(self, exc: Exception) -> None:
        self._last_error = str(exc)[:1000]
