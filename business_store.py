"""P1 business data: assets, managed knowledge, cases, procedures and metrics."""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from document_ingestion import chunk_text
from runtime_store import DEFAULT_DB_PATH, iso
from security import ROLE_LEVEL


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", str(text or "").lower())
    latin = set(re.findall(r"[a-z0-9_\-]{2,}", normalized))
    chinese_chars = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    chinese = {chinese_chars[index:index + 2] for index in range(len(chinese_chars) - 1)}
    return latin | chinese


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


class BusinessStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fts_available = False
        self.setup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def setup(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, asset_code TEXT NOT NULL,
                    name TEXT NOT NULL, asset_type TEXT NOT NULL DEFAULT '', vendor TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '', serial_no TEXT NOT NULL DEFAULT '', firmware TEXT NOT NULL DEFAULT '',
                    criticality TEXT NOT NULL DEFAULT 'medium', location TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, asset_code)
                );
                CREATE INDEX IF NOT EXISTS idx_assets_tenant ON assets(tenant_id, status, asset_type);

                CREATE TABLE IF NOT EXISTS asset_measurements (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, asset_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, name TEXT NOT NULL, numeric_value REAL,
                    text_value TEXT NOT NULL DEFAULT '', unit TEXT NOT NULL DEFAULT '',
                    captured_at TEXT NOT NULL, UNIQUE(session_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_asset_measurements
                    ON asset_measurements(tenant_id, asset_id, name, captured_at DESC);

                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, title TEXT NOT NULL,
                    source_filename TEXT NOT NULL, file_type TEXT NOT NULL, min_role TEXT NOT NULL DEFAULT 'viewer',
                    status TEXT NOT NULL DEFAULT 'active', current_version INTEGER NOT NULL DEFAULT 1,
                    owner_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_docs ON knowledge_documents(tenant_id, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS knowledge_versions (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version INTEGER NOT NULL,
                    checksum TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                    change_summary TEXT NOT NULL DEFAULT '', created_by TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(document_id, version), FOREIGN KEY(document_id) REFERENCES knowledge_documents(id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES knowledge_documents(id),
                    FOREIGN KEY(version_id) REFERENCES knowledge_versions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks ON knowledge_chunks(tenant_id, document_id, version_id);

                CREATE TABLE IF NOT EXISTS diagnosis_cases (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL DEFAULT '', fault_description TEXT NOT NULL,
                    confirmed_root_cause TEXT NOT NULL DEFAULT '', resolution TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT 'unverified', credibility REAL NOT NULL DEFAULT 0.5,
                    expert_rating INTEGER, status TEXT NOT NULL DEFAULT 'active', searchable_text TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_tenant ON diagnosis_cases(tenant_id, status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS procedure_versions (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
                    version INTEGER NOT NULL, mermaid TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                    author_id TEXT NOT NULL, approver_id TEXT NOT NULL DEFAULT '',
                    change_summary TEXT NOT NULL DEFAULT '', decision_note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, decided_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(session_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_procedures ON procedure_versions(tenant_id, session_id, version DESC);

                CREATE TABLE IF NOT EXISTS diagnosis_metrics (
                    session_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, asset_id TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '', input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0, estimated_cost REAL NOT NULL DEFAULT 0,
                    elapsed_ms INTEGER NOT NULL DEFAULT 0, search_count INTEGER NOT NULL DEFAULT 0,
                    source_count INTEGER NOT NULL DEFAULT 0, revision_count INTEGER NOT NULL DEFAULT 0,
                    expert_feedback_count INTEGER NOT NULL DEFAULT 0, audit_passed INTEGER NOT NULL DEFAULT 0,
                    mermaid_valid INTEGER NOT NULL DEFAULT 0, risk_level TEXT NOT NULL DEFAULT 'low',
                    created_at TEXT NOT NULL
                );
                """
            )
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(chunk_id UNINDEXED, tenant_id UNINDEXED, content, tokenize='unicode61')"
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False
            version_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(knowledge_versions)")
            }
            migrations = {
                "source_path": "TEXT NOT NULL DEFAULT ''",
                "index_status": "TEXT NOT NULL DEFAULT 'pending'",
                "index_backend": "TEXT NOT NULL DEFAULT ''",
                "index_error": "TEXT NOT NULL DEFAULT ''",
                "indexed_at": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in version_columns:
                    conn.execute(
                        f"ALTER TABLE knowledge_versions ADD COLUMN {name} {definition}"
                    )
            asset_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(assets)")
            }
            if "parent_id" not in asset_columns:
                conn.execute("ALTER TABLE assets ADD COLUMN parent_id TEXT NOT NULL DEFAULT ''")
            document_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(knowledge_documents)")
            }
            if "applicability" not in document_columns:
                conn.execute(
                    "ALTER TABLE knowledge_documents ADD COLUMN applicability TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("metadata", "applicability"):
            if key in result:
                result[key] = json.loads(result[key] or "{}")
        return result

    # Assets
    def create_asset(self, tenant_id: str, payload: dict) -> dict:
        asset_id, now = uuid.uuid4().hex, iso()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO assets(id, tenant_id, asset_code, name, asset_type, vendor, model,
                   serial_no, firmware, criticality, location, metadata, status, created_at, updated_at,
                   parent_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    asset_id, tenant_id, payload["asset_code"], payload["name"],
                    payload.get("asset_type", ""), payload.get("vendor", ""), payload.get("model", ""),
                    payload.get("serial_no", ""), payload.get("firmware", ""),
                    payload.get("criticality", "medium"), payload.get("location", ""),
                    _json(payload.get("metadata", {})), "active", now, now,
                    payload.get("parent_id", ""),
                ),
            )
        return self.get_asset(tenant_id, asset_id)

    def get_asset(self, tenant_id: str, asset_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE tenant_id=? AND id=?", (tenant_id, asset_id)
            ).fetchone()
        return self._row(row)

    def get_asset_by_code(self, tenant_id: str, asset_code: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE tenant_id=? AND asset_code=?",
                (tenant_id, asset_code),
            ).fetchone()
        return self._row(row)

    def list_assets(self, tenant_id: str, include_inactive: bool = False) -> list[dict]:
        sql = "SELECT * FROM assets WHERE tenant_id=?"
        params: list[Any] = [tenant_id]
        if not include_inactive:
            sql += " AND status='active'"
        sql += " ORDER BY asset_code"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row(row) for row in rows]

    def update_asset(self, tenant_id: str, asset_id: str, payload: dict) -> dict | None:
        allowed = {"asset_code", "name", "asset_type", "vendor", "model", "serial_no", "firmware", "criticality", "location", "metadata", "status", "parent_id"}
        updates = {key: value for key, value in payload.items() if key in allowed}
        if "metadata" in updates:
            updates["metadata"] = _json(updates["metadata"])
        updates["updated_at"] = iso()
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE assets SET {', '.join(f'{key}=?' for key in updates)} WHERE tenant_id=? AND id=?",
                [*updates.values(), tenant_id, asset_id],
            )
        return self.get_asset(tenant_id, asset_id) if cursor.rowcount else None

    def record_measurements(
        self, tenant_id: str, asset_id: str, session_id: str,
        measurements: dict, asset: dict | None = None,
    ) -> int:
        if not asset_id or not measurements:
            return 0
        template = ((asset or {}).get("metadata") or {}).get("measurement_template", {})
        now, rows = iso(), []
        for name, raw in measurements.items():
            spec = template.get(name, {}) if isinstance(template, dict) else {}
            if isinstance(raw, dict):
                value, unit = raw.get("value"), str(raw.get("unit") or spec.get("unit") or "")
            else:
                value, unit = raw, str(spec.get("unit") or "")
            numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            text_value = "" if numeric is not None else str(value or "")[:500]
            rows.append((uuid.uuid4().hex, tenant_id, asset_id, session_id, str(name), numeric, text_value, unit, now))
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO asset_measurements(
                   id,tenant_id,asset_id,session_id,name,numeric_value,text_value,unit,captured_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    # Managed knowledge
    def import_document(
        self, tenant_id: str, actor_id: str, title: str, filename: str, file_type: str,
        content: str, min_role: str = "viewer", document_id: str | None = None,
        change_summary: str = "", applicability: dict | None = None,
    ) -> dict:
        if min_role not in ROLE_LEVEL:
            raise ValueError("无效的知识权限角色")
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = iso()
        with self._lock, self._connect() as conn:
            if document_id:
                doc = conn.execute(
                    "SELECT * FROM knowledge_documents WHERE id=? AND tenant_id=?", (document_id, tenant_id)
                ).fetchone()
                if not doc:
                    raise KeyError("知识文档不存在")
                latest = conn.execute(
                    "SELECT checksum FROM knowledge_versions WHERE document_id=? ORDER BY version DESC LIMIT 1",
                    (document_id,),
                ).fetchone()
                if latest and latest["checksum"] == checksum:
                    raise ValueError("文档内容与当前版本相同，无需重复索引")
                version = int(doc["current_version"]) + 1
                conn.execute(
                    "UPDATE knowledge_documents SET title=?, source_filename=?, file_type=?, min_role=?, applicability=?, current_version=?, status='active', updated_at=? WHERE id=?",
                    (title, filename, file_type, min_role, _json(applicability or {}), version, now, document_id),
                )
                conn.execute(
                    "UPDATE knowledge_versions SET status='superseded' WHERE document_id=? AND status='active'",
                    (document_id,),
                )
            else:
                document_id, version = uuid.uuid4().hex, 1
                conn.execute(
                    """INSERT INTO knowledge_documents(
                       id, tenant_id, title, source_filename, file_type, min_role,
                       status, current_version, owner_id, created_at, updated_at, applicability
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (document_id, tenant_id, title, filename, file_type, min_role, "active", version, actor_id, now, now, _json(applicability or {})),
                )
            version_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO knowledge_versions(
                   id, document_id, version, checksum, content, status,
                   change_summary, created_by, created_at, index_status
                   ) VALUES(?,?,?,?,?,?,?,?,?,'pending')""",
                (version_id, document_id, version, checksum, content, "active", change_summary, actor_id, now),
            )
            chunks = chunk_text(content)
            for index, item in enumerate(chunks):
                chunk_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO knowledge_chunks VALUES(?,?,?,?,?,?,?,?)",
                    (chunk_id, document_id, version_id, tenant_id, index, item["content"], item["location"], "{}"),
                )
                if self._fts_available:
                    conn.execute(
                        "INSERT INTO knowledge_fts(chunk_id, tenant_id, content) VALUES(?,?,?)",
                        (chunk_id, tenant_id, item["content"]),
                    )
        return {
            "document_id": document_id, "version_id": version_id,
            "version": version, "chunks": len(chunks), "checksum": checksum,
            "index_status": "pending",
        }

    def list_documents(self, tenant_id: str, role: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT d.*, v.id AS current_version_id,
                          v.source_path, v.index_status, v.index_backend,
                          v.index_error, v.indexed_at
                   FROM knowledge_documents d
                   LEFT JOIN knowledge_versions v
                     ON v.document_id=d.id AND v.version=d.current_version
                   WHERE d.tenant_id=? ORDER BY d.updated_at DESC""",
                (tenant_id,),
            ).fetchall()
        level = ROLE_LEVEL.get(role, 0)
        result = []
        for row in rows:
            if level < ROLE_LEVEL.get(row["min_role"], 999):
                continue
            item = dict(row)
            item["applicability"] = json.loads(item.get("applicability") or "{}")
            result.append(item)
        return result

    def document_versions(self, tenant_id: str, document_id: str, role: str) -> list[dict]:
        docs = {item["id"]: item for item in self.list_documents(tenant_id, role)}
        if document_id not in docs:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, version, checksum, status, change_summary,
                          created_by, created_at, source_path, index_status,
                          index_backend, index_error, indexed_at
                   FROM knowledge_versions
                   WHERE document_id=? ORDER BY version DESC""",
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def document_for_indexing(self, tenant_id: str, document_id: str) -> dict | None:
        """Return the current version and its chunks for rebuilding Milvus."""
        with self._connect() as conn:
            doc = conn.execute(
                """SELECT d.*, v.id AS version_id, v.version, v.index_status
                   FROM knowledge_documents d
                   JOIN knowledge_versions v
                     ON v.document_id=d.id AND v.version=d.current_version
                   WHERE d.tenant_id=? AND d.id=?""",
                (tenant_id, document_id),
            ).fetchone()
            if not doc:
                return None
            chunks = conn.execute(
                """SELECT id, chunk_index, content, location
                   FROM knowledge_chunks WHERE version_id=? ORDER BY chunk_index""",
                (doc["version_id"],),
            ).fetchall()
        return {
            **dict(doc), "document_id": doc["id"],
            "document_status": doc["status"],
            "chunks": [dict(row) for row in chunks],
        }

    def update_version_artifacts(
        self, version_id: str, *, source_path: str | None = None,
        index_status: str | None = None, index_backend: str | None = None,
        index_error: str | None = None,
    ) -> None:
        updates: dict[str, str] = {}
        if source_path is not None:
            updates["source_path"] = source_path
        if index_status is not None:
            updates["index_status"] = index_status
            updates["indexed_at"] = iso() if index_status in {"ready", "removed"} else ""
        if index_backend is not None:
            updates["index_backend"] = index_backend
        if index_error is not None:
            updates["index_error"] = index_error[:2000]
        if not updates:
            return
        with self._connect() as conn:
            conn.execute(
                f"UPDATE knowledge_versions SET {', '.join(f'{key}=?' for key in updates)} WHERE id=?",
                [*updates.values(), version_id],
            )

    def set_document_status(self, tenant_id: str, document_id: str, status: str) -> bool:
        if status not in {"active", "inactive", "archived"}:
            raise ValueError("无效的文档状态")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE knowledge_documents SET status=?, updated_at=? WHERE tenant_id=? AND id=?",
                (status, iso(), tenant_id, document_id),
            )
        return cursor.rowcount > 0

    def search_knowledge(
        self, tenant_id: str, role: str, query: str, limit: int = 5,
        allowed_document_ids: set[str] | None = None,
    ) -> list[dict]:
        accessible = {item["id"]: item for item in self.list_documents(tenant_id, role) if item["status"] == "active"}
        if not accessible:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.*, d.title, d.current_version, d.min_role
                   FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id
                   JOIN knowledge_versions v ON v.id=c.version_id
                   WHERE c.tenant_id=? AND d.status='active' AND v.status='active'""",
                (tenant_id,),
            ).fetchall()
        scored = []
        for row in rows:
            if row["document_id"] not in accessible:
                continue
            if allowed_document_ids is not None and row["document_id"] not in allowed_document_ids:
                continue
            score = _similarity(query, row["content"])
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "evidence_id": f"sop:{row['id']}", "source_type": "managed_sop",
                "document_id": row["document_id"], "title": row["title"],
                "version": row["current_version"], "location": row["location"],
                "content": row["content"], "score": round(score, 4), "trust_level": "authoritative",
            }
            for score, row in scored[: max(1, min(limit, 20))]
        ]

    def asset_detail(self, tenant_id: str, asset_id: str) -> dict | None:
        asset = self.get_asset(tenant_id, asset_id)
        if not asset:
            return None
        with self._connect() as conn:
            has_jobs = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='diagnosis_jobs'"
            ).fetchone()
            job_rows = (
                conn.execute(
                    """SELECT session_id, status, fault_input, pending_kind, error_code,
                              request_payload, created_at, updated_at
                       FROM diagnosis_jobs WHERE tenant_id=? ORDER BY updated_at DESC LIMIT 500""",
                    (tenant_id,),
                ).fetchall()
                if has_jobs else []
            )
            cases = conn.execute(
                """SELECT session_id, fault_description, confirmed_root_cause,
                          resolution, outcome, credibility, expert_rating, updated_at
                   FROM diagnosis_cases WHERE tenant_id=? AND asset_id=?
                   ORDER BY updated_at DESC LIMIT 100""",
                (tenant_id, asset_id),
            ).fetchall()
            metrics = conn.execute(
                """SELECT COUNT(*) jobs, COALESCE(AVG(elapsed_ms),0) avg_elapsed_ms,
                          COALESCE(AVG(audit_passed),0) audit_pass_rate,
                          COALESCE(SUM(estimated_cost),0) cost
                   FROM diagnosis_metrics WHERE tenant_id=? AND asset_id=?""",
                (tenant_id, asset_id),
            ).fetchone()
            measurements = conn.execute(
                """SELECT session_id,name,numeric_value,text_value,unit,captured_at
                   FROM asset_measurements WHERE tenant_id=? AND asset_id=?
                   ORDER BY captured_at DESC LIMIT 500""",
                (tenant_id, asset_id),
            ).fetchall()
            trends = conn.execute(
                """SELECT name,unit,COUNT(*) samples,MIN(numeric_value) minimum,
                          MAX(numeric_value) maximum,AVG(numeric_value) average,
                          MAX(captured_at) latest_at
                   FROM asset_measurements
                   WHERE tenant_id=? AND asset_id=? AND numeric_value IS NOT NULL
                   GROUP BY name,unit ORDER BY name""",
                (tenant_id, asset_id),
            ).fetchall()
        jobs = []
        for row in job_rows:
            payload = json.loads(row["request_payload"] or "{}")
            if str(payload.get("asset_id") or "") != asset_id:
                continue
            item = dict(row)
            item.pop("request_payload", None)
            item["alarm_code"] = payload.get("alarm_code", "")
            item["operating_context"] = payload.get("operating_context", "")
            jobs.append(item)
        return {
            "asset": asset, "jobs": jobs[:100],
            "cases": [dict(row) for row in cases], "metrics": dict(metrics),
            "measurements": [dict(row) for row in measurements],
            "measurement_trends": [dict(row) for row in trends],
        }

    # Cases
    def save_case(self, session_id: str, tenant_id: str, asset_id: str, state: dict) -> dict:
        now, case_id = iso(), uuid.uuid4().hex
        fault = state.get("fault_input", "")
        searchable = "\n".join([
            fault, state.get("filtered_context", ""), state.get("audit_result", ""),
            state.get("mermaid_diagram", ""),
        ])[:20000]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO diagnosis_cases(id, session_id, tenant_id, asset_id, fault_description,
                   searchable_text, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET searchable_text=excluded.searchable_text, updated_at=excluded.updated_at""",
                (case_id, session_id, tenant_id, asset_id or "", fault, searchable, now, now),
            )
        return self.get_case_by_session(tenant_id, session_id)

    def get_case_by_session(self, tenant_id: str, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM diagnosis_cases WHERE tenant_id=? AND session_id=?", (tenant_id, session_id)
            ).fetchone()
        return dict(row) if row else None

    def confirm_case(self, tenant_id: str, session_id: str, payload: dict) -> dict | None:
        rating = payload.get("expert_rating")
        credibility = min(1.0, max(0.0, float(payload.get("credibility", 0.9))))
        with self._connect() as conn:
            current = conn.execute(
                "SELECT searchable_text FROM diagnosis_cases WHERE tenant_id=? AND session_id=?",
                (tenant_id, session_id),
            ).fetchone()
            if not current:
                return None
            searchable = "\n".join(filter(None, [
                current["searchable_text"], payload.get("confirmed_root_cause", ""),
                payload.get("resolution", ""), payload.get("outcome", ""),
            ]))[:24000]
            conn.execute(
                """UPDATE diagnosis_cases SET confirmed_root_cause=?, resolution=?, outcome=?, credibility=?,
                   expert_rating=?, searchable_text=?, updated_at=? WHERE tenant_id=? AND session_id=?""",
                (
                    payload.get("confirmed_root_cause", ""), payload.get("resolution", ""),
                    payload.get("outcome", "resolved"), credibility, rating, searchable,
                    iso(), tenant_id, session_id,
                ),
            )
        return self.get_case_by_session(tenant_id, session_id)

    def search_cases(self, tenant_id: str, query: str, asset_id: str = "", limit: int = 3) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM diagnosis_cases WHERE tenant_id=? AND status='active' ORDER BY updated_at DESC LIMIT 500",
                (tenant_id,),
            ).fetchall()
        scored = []
        for row in rows:
            score = _similarity(query, row["searchable_text"])
            if asset_id and row["asset_id"] == asset_id:
                score += 0.2
            score *= 0.5 + float(row["credibility"]) * 0.5
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**dict(row), "score": round(score, 4), "evidence_id": f"case:{row['id']}", "source_type": "historical_case"}
            for score, row in scored[: max(1, min(limit, 20))]
        ]

    # Procedure lifecycle
    def create_procedure_version(self, session_id: str, tenant_id: str, author_id: str, mermaid: str, change_summary: str = "") -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) AS value FROM procedure_versions WHERE session_id=? AND tenant_id=?",
                (session_id, tenant_id),
            ).fetchone()
            version = int(row["value"] or 0) + 1
            version_id, now = uuid.uuid4().hex, iso()
            conn.execute(
                "INSERT INTO procedure_versions VALUES(?,?,?,?,?,'draft',?,'',?,'',?,'')",
                (version_id, session_id, tenant_id, version, mermaid, author_id, change_summary, now),
            )
        return self.procedure_version(tenant_id, version_id)

    def procedure_version(self, tenant_id: str, version_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM procedure_versions WHERE tenant_id=? AND id=?", (tenant_id, version_id)
            ).fetchone()
        return dict(row) if row else None

    def list_procedures(self, tenant_id: str, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM procedure_versions WHERE tenant_id=? AND session_id=? ORDER BY version DESC",
                (tenant_id, session_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def procedure_diff(self, tenant_id: str, left_id: str, right_id: str) -> str:
        left, right = self.procedure_version(tenant_id, left_id), self.procedure_version(tenant_id, right_id)
        if not left or not right:
            raise KeyError("流程版本不存在")
        return "\n".join(difflib.unified_diff(
            left["mermaid"].splitlines(), right["mermaid"].splitlines(),
            fromfile=f"v{left['version']}", tofile=f"v{right['version']}", lineterm="",
        ))

    def procedure_diff_hunks(self, tenant_id: str, left_id: str, right_id: str) -> list[dict]:
        left, right = self.procedure_version(tenant_id, left_id), self.procedure_version(tenant_id, right_id)
        if not left or not right:
            raise KeyError("流程版本不存在")
        left_lines, right_lines = left["mermaid"].splitlines(), right["mermaid"].splitlines()
        matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines)
        hunks = []
        for opcode_index, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes()):
            if tag == "equal":
                continue
            hunks.append({
                "id": opcode_index, "operation": tag,
                "left_start": i1 + 1, "right_start": j1 + 1,
                "before": left_lines[i1:i2], "after": right_lines[j1:j2],
            })
        return hunks

    def adopt_procedure_hunks(
        self, tenant_id: str, left_id: str, right_id: str, accepted_hunks: list[int]
    ) -> str:
        left, right = self.procedure_version(tenant_id, left_id), self.procedure_version(tenant_id, right_id)
        if not left or not right:
            raise KeyError("流程版本不存在")
        if left["session_id"] != right["session_id"]:
            raise ValueError("只能采纳同一任务的流程差异")
        accepted = {int(item) for item in accepted_hunks}
        left_lines, right_lines = left["mermaid"].splitlines(), right["mermaid"].splitlines()
        matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines)
        result = []
        valid_ids = set()
        for opcode_index, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes()):
            if tag == "equal":
                result.extend(left_lines[i1:i2])
                continue
            valid_ids.add(opcode_index)
            result.extend(right_lines[j1:j2] if opcode_index in accepted else left_lines[i1:i2])
        if not accepted.issubset(valid_ids):
            raise ValueError("包含不存在的差异项")
        return "\n".join(result)

    def decide_procedure(self, tenant_id: str, version_id: str, actor_id: str, decision: str, note: str = "") -> dict | None:
        if decision not in {"approved", "rejected", "published"}:
            raise ValueError("无效审批决定")
        with self._connect() as conn:
            target = conn.execute(
                "SELECT * FROM procedure_versions WHERE tenant_id=? AND id=?", (tenant_id, version_id)
            ).fetchone()
            if not target:
                return None
            current_status = target["status"]
            if decision == "approved" and current_status != "draft":
                raise ValueError("只有草稿版本可以批准")
            if decision == "rejected" and current_status not in {"draft", "approved"}:
                raise ValueError("当前版本不可拒绝")
            if decision == "published" and current_status != "approved":
                raise ValueError("流程版本必须先批准才能发布")
            if decision == "published":
                conn.execute(
                    "UPDATE procedure_versions SET status='superseded' WHERE tenant_id=? AND session_id=? AND status='published'",
                    (tenant_id, target["session_id"]),
                )
            conn.execute(
                "UPDATE procedure_versions SET status=?, approver_id=?, decision_note=?, decided_at=? WHERE tenant_id=? AND id=?",
                (decision, actor_id, note, iso(), tenant_id, version_id),
            )
        return self.procedure_version(tenant_id, version_id)

    # Metrics/dashboard
    def record_metrics(self, session_id: str, tenant_id: str, asset_id: str, state: dict, elapsed_ms: int) -> None:
        usage = state.get("usage_events", []) or []
        input_tokens = sum(int(item.get("input_tokens", 0)) for item in usage)
        output_tokens = sum(int(item.get("output_tokens", 0)) for item in usage)
        input_rate = float(__import__("os").environ.get("LLM_INPUT_COST_PER_1M", "0"))
        output_rate = float(__import__("os").environ.get("LLM_OUTPUT_COST_PER_1M", "0"))
        cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
        external = state.get("external_result", {}) or {}
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO diagnosis_metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, tenant_id, asset_id or "", __import__("os").environ.get("LLM_MODEL", ""),
                    input_tokens, output_tokens, cost, elapsed_ms,
                    int(external.get("search_count", 0)), len(external.get("sources", [])),
                    int(state.get("revision_count", 0)), len(state.get("expert_feedbacks", [])),
                    int(not state.get("has_gaps", True)), int(bool((state.get("mermaid_validation") or {}).get("valid"))),
                    (state.get("safety_assessment") or {}).get("risk_level", "low"), iso(),
                ),
            )

    def dashboard(self, tenant_id: str) -> dict:
        with self._connect() as conn:
            summary = conn.execute(
                """SELECT COUNT(*) jobs, COALESCE(SUM(estimated_cost),0) cost,
                   COALESCE(AVG(elapsed_ms),0) avg_elapsed_ms, COALESCE(AVG(audit_passed),0) audit_pass_rate,
                   COALESCE(AVG(revision_count),0) avg_revisions, COALESCE(SUM(search_count),0) searches,
                   COALESCE(SUM(input_tokens+output_tokens),0) tokens,
                   COALESCE(AVG(CASE WHEN expert_feedback_count>0 OR revision_count>0 THEN 1.0 ELSE 0.0 END),0) expert_modification_rate
                   FROM diagnosis_metrics WHERE tenant_id=?""",
                (tenant_id,),
            ).fetchone()
            case_summary = conn.execute(
                """SELECT COUNT(*) confirmed_cases,
                   COALESCE(AVG(CASE WHEN outcome='resolved' THEN 1.0 ELSE 0.0 END),0) case_success_rate
                   FROM diagnosis_cases WHERE tenant_id=? AND outcome!='unverified'""",
                (tenant_id,),
            ).fetchone()
            by_asset = conn.execute(
                """SELECT asset_id, COUNT(*) jobs, AVG(audit_passed) audit_pass_rate,
                   AVG(elapsed_ms) avg_elapsed_ms, SUM(estimated_cost) cost
                   FROM diagnosis_metrics WHERE tenant_id=? GROUP BY asset_id ORDER BY jobs DESC LIMIT 20""",
                (tenant_id,),
            ).fetchall()
            by_model = conn.execute(
                """SELECT model, COUNT(*) jobs, SUM(input_tokens+output_tokens) tokens,
                   SUM(estimated_cost) cost FROM diagnosis_metrics WHERE tenant_id=? GROUP BY model""",
                (tenant_id,),
            ).fetchall()
        summary_result = dict(summary)
        summary_result.update(dict(case_summary))
        return {"summary": summary_result, "by_asset": [dict(row) for row in by_asset], "by_model": [dict(row) for row in by_model]}

    def health(self) -> dict:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM assets LIMIT 1").fetchone()
            return {"ok": True, "backend": "sqlite", "fts": self._fts_available}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
