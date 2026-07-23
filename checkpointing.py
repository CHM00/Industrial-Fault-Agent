"""Checkpoint backend selection for development and pilot deployments."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "output" / "runtime" / "checkpoints.sqlite3"


def configured_backend() -> str:
    default = "sqlite" if os.environ.get("APP_ENV", "development").lower() == "production" else "memory"
    return os.environ.get("CHECKPOINT_BACKEND", default).strip().lower()


def create_checkpointer(backend: str | None = None):
    """Create a LangGraph checkpointer.

    SQLite is intentionally provided by LangGraph's official integration package.
    Production never silently falls back to volatile memory.
    """
    selected = (backend or configured_backend()).lower()
    if selected == "memory":
        return MemorySaver()
    if selected != "sqlite":
        raise RuntimeError(f"不支持的 CHECKPOINT_BACKEND: {selected}")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpoint 后端未安装，请安装 langgraph-checkpoint-sqlite==3.1.0"
        ) from exc

    path = Path(os.environ.get("CHECKPOINT_SQLITE_PATH", str(DEFAULT_CHECKPOINT_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")
    saver = SqliteSaver(connection)
    saver.setup()
    # Keep the connection reachable for the lifetime of the compiled graph.
    saver._pilot_connection = connection
    saver._pilot_backend = "sqlite"
    saver._pilot_path = str(path)
    return saver


def checkpointer_status(checkpointer) -> dict:
    backend = getattr(checkpointer, "_pilot_backend", "memory")
    result = {"backend": backend, "ok": True}
    if backend == "sqlite":
        try:
            connection = getattr(checkpointer, "_pilot_connection")
            connection.execute("SELECT 1").fetchone()
            result["path"] = getattr(checkpointer, "_pilot_path", "")
        except Exception as exc:
            result.update({"ok": False, "error": str(exc)})
    return result
