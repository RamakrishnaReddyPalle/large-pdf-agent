# src/agent/logger.py
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Optional
from .config import CFG

DDL = {
    "conversations": """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  created_ts REAL
);""",
    "messages": """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  ts REAL,
  role TEXT,
  content TEXT
);""",
    "events": """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL,
  event TEXT,
  payload TEXT
);"""
}

REQUIRED_COLS = {
    "conversations": ["id", "created_ts"],
    "messages": ["id", "session_id", "ts", "role", "content"],
    "events": ["id", "ts", "event", "payload"],
}

class EventLogger:
    """
    Minimal SQLite logger with self-healing schema:
      conversations(id, created_ts)
      messages(id, session_id, ts, role, content)
      events(id, ts, event, payload)
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or CFG.sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        return self.db_path

    def _exec(self, cur, sql: str):
        cur.execute(sql)

    def _table_cols(self, cur, table: str) -> list[str]:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return [r[1] for r in cur.fetchall()]
        except Exception:
            return []

    def _ensure_table(self, cur, table: str):
        # Create if not exists
        self._exec(cur, DDL[table])
        # Check columns
        existing = self._table_cols(cur, table)
        if not existing:
            return

        need = set(REQUIRED_COLS[table])
        have = set(existing)

        # Fast path: already good
        if need.issubset(have):
            return

        # Try to add missing columns if names align
        addable = need - have
        safe_add = {"ts", "event", "payload", "role", "content", "session_id", "created_ts"}  # common adds
        if addable and addable.issubset(safe_add):
            for col in addable:
                if col == "ts":
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN ts REAL")
                elif col == "created_ts":
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN created_ts REAL")
                elif col in {"event", "payload", "role", "content", "session_id"}:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            return

        # If incompatible (e.g., old names), migrate by rename → create → copy what we can
        cur.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        self._exec(cur, DDL[table])

        old_cols = self._table_cols(cur, f"{table}_old")

        # Copy best-effort per table
        if table == "conversations":
            # old might have: session_id / created
            src_id = "id" if "id" in old_cols else ("session_id" if "session_id" in old_cols else None)
            src_created = "created_ts" if "created_ts" in old_cols else ("created" if "created" in old_cols else None)
            if src_id:
                cur.execute(f"INSERT OR IGNORE INTO conversations(id, created_ts) "
                            f"SELECT {src_id}, COALESCE({src_created}, strftime('%s','now')) FROM {table}_old")
        elif table == "messages":
            # old might miss ts; or have created instead
            cols_map = {
                "session_id": "session_id" if "session_id" in old_cols else None,
                "ts": "ts" if "ts" in old_cols else ("created" if "created" in old_cols else None),
                "role": "role" if "role" in old_cols else None,
                "content": "content" if "content" in old_cols else None,
            }
            if all(cols_map.values()):
                cur.execute(
                    f"INSERT INTO messages(session_id, ts, role, content) "
                    f"SELECT {cols_map['session_id']}, {cols_map['ts']}, {cols_map['role']}, {cols_map['content']} "
                    f"FROM {table}_old"
                )
        elif table == "events":
            # old might miss ts
            cols_map = {
                "ts": "ts" if "ts" in old_cols else ("time" if "time" in old_cols else None),
                "event": "event" if "event" in old_cols else None,
                "payload": "payload" if "payload" in old_cols else None,
            }
            if all(cols_map.values()):
                cur.execute(
                    f"INSERT INTO events(ts, event, payload) "
                    f"SELECT {cols_map['ts']}, {cols_map['event']}, {cols_map['payload']} FROM {table}_old"
                )
        # Keep old table for audit
        # If you want to drop it after migration: cur.execute(f"DROP TABLE {table}_old")

    def _init_schema(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        for t in ("conversations", "messages", "events"):
            self._ensure_table(cur, t)
        con.commit()
        con.close()

    def add_conv_if_missing(self, session_id: str):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("INSERT OR IGNORE INTO conversations(id, created_ts) VALUES(?, ?)",
                    (session_id, time.time()))
        con.commit()
        con.close()

    def log_msg(self, session_id: str, role: str, content: str):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, ts, role, content) VALUES (?,?,?,?)",
            (session_id, time.time(), role, content),
        )
        con.commit()
        con.close()

    def log_event(self, event: str, payload: dict | None = None):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO events(ts, event, payload) VALUES (?,?,?)",
            (time.time(), event, json.dumps(payload or {})),
        )
        con.commit()
        con.close()
