"""SQLite (WAL) request telemetry. One connection per process, schema initialized once."""
import datetime
import sqlite3
import threading
from typing import Any, Dict, Optional

import pandas as pd

from semantic_rag.settings import get_settings

_COLUMNS = {
    "timestamp": "TEXT NOT NULL", "query": "TEXT NOT NULL", "model": "TEXT NOT NULL",
    "prompt_tokens": "INTEGER DEFAULT 0", "completion_tokens": "INTEGER DEFAULT 0", "cost": "REAL DEFAULT 0.0",
    "latency_ms": "REAL DEFAULT 0.0", "cache_hit": "INTEGER DEFAULT 0", "route": "TEXT NOT NULL",
    "escalated": "INTEGER DEFAULT 0",
    # added in v3
    "status": "TEXT", "tenant_id": "TEXT", "domain": "TEXT", "primary_document": "TEXT",
    "top_relevance": "REAL", "embed_ms": "REAL", "cache_ms": "REAL", "search_ms": "REAL", "rerank_ms": "REAL",
    "generation_ms": "REAL", "error": "TEXT",
}

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                path = get_settings().telemetry_db
                path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                             + ", ".join(f"{k} {v}" for k, v in list(_COLUMNS.items())[:10]) + ")")
                existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
                for col, decl in _COLUMNS.items():
                    if col not in existing:  # migrate databases created by older versions
                        conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {decl}")
                conn.commit()
                _conn = conn
    return _conn


def init_db() -> None:
    _connection()


def log_request(record: Dict[str, Any]) -> None:
    row = {k: record.get(k) for k in _COLUMNS if k != "timestamp"}
    row["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for k in ("cache_hit", "escalated"):
        row[k] = 1 if row.get(k) else 0
    cols = list(row.keys())
    conn = _connection()
    with _lock:
        conn.execute(f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                     [row[c] for c in cols])
        conn.commit()


def load_requests(limit: int = 5000) -> pd.DataFrame:
    conn = _connection()
    with _lock:
        return pd.read_sql(f"SELECT * FROM requests ORDER BY id DESC LIMIT {int(limit)}", conn)
