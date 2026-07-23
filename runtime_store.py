"""Durable diagnosis job/session registry backed by SQLite."""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "output" / "runtime" / "pilot.sqlite3"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(value: dt.datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


@dataclass
class JobRecord:
    session_id: str
    request_id: str
    thread_id: str
    owner_id: str
    tenant_id: str
    status: str
    fault_input: str
    auto_mode: bool
    pending_kind: str = ""
    request_payload: dict[str, Any] | None = None
    result_paths: dict[str, Any] | None = None
    error_code: str = ""
    created_at: str = ""
    updated_at: str = ""
    expires_at: str = ""

    def public_dict(self) -> dict:
        value = asdict(self)
        payload = value.pop("request_payload", None) or {}
        value["asset_id"] = str(payload.get("asset_id") or "")
        value["alarm_code"] = str(payload.get("alarm_code") or "")
        value["operating_context"] = str(payload.get("operating_context") or "")
        return value


class JobStore:
    def __init__(self, path: str | Path | None = None, ttl_seconds: int | None = None):
        self.path = Path(path or os.environ.get("PILOT_DB_PATH", str(DEFAULT_DB_PATH)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds or int(os.environ.get("SESSION_TTL_SECONDS", "86400"))
        self._schema_lock = threading.Lock()
        self.setup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def setup(self) -> None:
        with self._schema_lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_jobs (
                    session_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fault_input TEXT NOT NULL,
                    auto_mode INTEGER NOT NULL DEFAULT 0,
                    pending_kind TEXT NOT NULL DEFAULT '',
                    request_payload TEXT NOT NULL DEFAULT '{}',
                    result_paths TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_owner_status
                    ON diagnosis_jobs(tenant_id, owner_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_expiry
                    ON diagnosis_jobs(expires_at);
                CREATE TABLE IF NOT EXISTS diagnosis_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON diagnosis_events(session_id, id);
                """
            )

    def create(self, record: JobRecord) -> JobRecord:
        now = utc_now()
        record.created_at = record.created_at or iso(now)
        record.updated_at = record.updated_at or iso(now)
        record.expires_at = record.expires_at or iso(now + dt.timedelta(seconds=self.ttl_seconds))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO diagnosis_jobs (
                    session_id, request_id, thread_id, owner_id, tenant_id, status,
                    fault_input, auto_mode, pending_kind, request_payload, result_paths,
                    error_code, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id, record.request_id, record.thread_id,
                    record.owner_id, record.tenant_id, record.status,
                    record.fault_input, int(record.auto_mode), record.pending_kind,
                    json.dumps(record.request_payload or {}, ensure_ascii=False),
                    json.dumps(record.result_paths or {}, ensure_ascii=False),
                    record.error_code, record.created_at, record.updated_at, record.expires_at,
                ),
            )
        self.add_event(record.session_id, record.owner_id, "created", {"status": record.status})
        return record

    def _from_row(self, row: sqlite3.Row | None) -> JobRecord | None:
        if row is None:
            return None
        data = dict(row)
        data["auto_mode"] = bool(data["auto_mode"])
        data["request_payload"] = json.loads(data["request_payload"] or "{}")
        data["result_paths"] = json.loads(data["result_paths"] or "{}")
        return JobRecord(**data)

    def get(self, session_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM diagnosis_jobs WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._from_row(row)

    def update(self, session_id: str, actor_id: str = "system", **fields) -> JobRecord | None:
        allowed = {"status", "pending_kind", "result_paths", "error_code", "expires_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return self.get(session_id)
        if "result_paths" in updates:
            updates["result_paths"] = json.dumps(updates["result_paths"] or {}, ensure_ascii=False)
        updates["updated_at"] = iso()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [session_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE diagnosis_jobs SET {assignments} WHERE session_id = ?", values)
        record = self.get(session_id)
        self.add_event(session_id, actor_id, "updated", fields)
        return record

    def list_jobs(self, tenant_id: str, owner_id: str | None = None, limit: int = 100) -> list[JobRecord]:
        sql = "SELECT * FROM diagnosis_jobs WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if owner_id:
            sql += " AND owner_id = ?"
            params.append(owner_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(limit, 5000)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows if row is not None]

    def add_event(self, session_id: str, actor_id: str, event_type: str, detail: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO diagnosis_events(session_id, actor_id, event_type, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, actor_id, event_type, json.dumps(detail or {}, ensure_ascii=False), iso()),
            )

    def events(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT actor_id, event_type, detail, created_at FROM diagnosis_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {**dict(row), "detail": json.loads(row["detail"] or "{}")}
            for row in rows
        ]

    def expire_stale(self) -> list[str]:
        now = iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM diagnosis_jobs WHERE expires_at < ? AND status NOT IN ('completed','pending_approval','archived','failed','cancelled','denied','expired')",
                (now,),
            ).fetchall()
            ids = [row["session_id"] for row in rows]
            if ids:
                conn.executemany(
                    "UPDATE diagnosis_jobs SET status='expired', updated_at=? WHERE session_id=?",
                    [(now, session_id) for session_id in ids],
                )
        return ids

    def reconcile_interrupted(self) -> list[str]:
        """Mark work that was running when this process last stopped."""
        now = iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM diagnosis_jobs WHERE status IN ('running','cancel_requested')"
            ).fetchall()
            ids = [row["session_id"] for row in rows]
            if ids:
                conn.executemany(
                    "UPDATE diagnosis_jobs SET status='interrupted', error_code='PROCESS_RESTART', updated_at=? WHERE session_id=?",
                    [(now, session_id) for session_id in ids],
                )
        for session_id in ids:
            self.add_event(session_id, "system", "process_restart", {"status": "interrupted"})
        return ids

    def health(self) -> dict:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return {"ok": True, "backend": "sqlite", "path": str(self.path)}
        except Exception as exc:
            return {"ok": False, "backend": "sqlite", "error": str(exc)}
