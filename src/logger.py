import os
import sqlite3
import datetime
from typing import Dict, Any, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "request_logs.db"))

def init_db():
    """Ensure SQLite database and requests table exist with WAL mode and 30s timeout for concurrency."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            cache_hit INTEGER DEFAULT 0,
            route TEXT NOT NULL,
            escalated INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def log_request(
    query: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0.0,
    latency_ms: float = 0.0,
    cache_hit: bool = False,
    route: str = "small",
    escalated: bool = False
):
    """Insert a new query log entry into the database with concurrency safety."""
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    cursor.execute("""
        INSERT INTO requests (
            timestamp, query, model, prompt_tokens, completion_tokens,
            cost, latency_ms, cache_hit, route, escalated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp_str,
        query,
        model,
        prompt_tokens,
        completion_tokens,
        cost,
        round(latency_ms, 2),
        1 if cache_hit else 0,
        route,
        1 if escalated else 0
    ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized with WAL mode at {DB_PATH}")
