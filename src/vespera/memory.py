"""Bounded frozen memory files and a SQLite FTS5 session store.

Two kinds of memory live here:

* ``MEMORY.md`` / ``USER.md`` -- small, curated Markdown files. They are read
  exactly once when a :class:`MemoryStore` is created and the resulting
  snapshot is used for the whole session, so the system prompt prefix stays
  byte-identical between turns and provider-side prompt caching keeps working.
  Updates are written to disk immediately but only show up in the *next*
  session's snapshot.
* ``sessions.db`` -- every message of every session, indexed with FTS5 so the
  agent can search old conversations instead of carrying them in context.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import MEMORY_CHAR_LIMIT, USER_CHAR_LIMIT, Config

TARGETS = ("memory", "user")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    raw_json    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SearchHit:
    session_id: str
    title: str
    role: str
    snippet: str
    created_at: str


def _searchable_text(message: dict[str, Any]) -> str:
    """Flatten a chat message into the text that should be full-text indexed."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):  # multi-part content
        parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        parts.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")
    return "\n".join(p for p in parts if p)


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 expression (quoted terms joined by OR)."""
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in terms)


class MemoryStore:
    """Frozen memory snapshot plus persistent, searchable session history."""

    limits = {"memory": MEMORY_CHAR_LIMIT, "user": USER_CHAR_LIMIT}

    def __init__(self, config: Config):
        self.config = config
        config.ensure_dirs()
        self._paths = {"memory": config.memory_file, "user": config.user_file}
        # Frozen once, never re-read during this session.
        self._snapshot = {t: self._read_capped(t) for t in TARGETS}
        self.conn = sqlite3.connect(str(config.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ files
    def _read_capped(self, target: str) -> str:
        path = self._paths[target]
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[: self.limits[target]]

    @property
    def memory_snapshot(self) -> str:
        return self._snapshot["memory"]

    @property
    def user_snapshot(self) -> str:
        return self._snapshot["user"]

    def read_live(self, target: str) -> str:
        """Current on-disk content (may differ from the frozen snapshot)."""
        self._check_target(target)
        path = self._paths[target]
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def update_file(self, target: str, content: str, mode: str = "replace") -> str:
        """Write ``content`` to MEMORY.md or USER.md, enforcing the char cap.

        ``mode="append"`` adds ``content`` as a new line after the existing
        text. Raises ``ValueError`` if the result would exceed the limit.
        """
        self._check_target(target)
        if mode not in ("replace", "append"):
            raise ValueError(f"mode must be 'replace' or 'append', got {mode!r}")
        if mode == "append":
            current = self.read_live(target).rstrip("\n")
            new_text = f"{current}\n{content.strip()}\n" if current else f"{content.strip()}\n"
        else:
            new_text = content
        limit = self.limits[target]
        if len(new_text) > limit:
            raise ValueError(
                f"{target} would be {len(new_text)} chars; the limit is {limit}. "
                "Condense or replace existing entries instead."
            )
        self._paths[target].write_text(new_text, encoding="utf-8")
        return new_text

    def usage(self) -> dict[str, dict[str, int]]:
        """Character usage for the live files and the frozen snapshot."""
        return {
            t: {
                "live": len(self.read_live(t)),
                "snapshot": len(self._snapshot[t]),
                "limit": self.limits[t],
            }
            for t in TARGETS
        }

    @staticmethod
    def _check_target(target: str) -> None:
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    # --------------------------------------------------------------- sessions
    def new_session(self, title: str = "") -> str:
        session_id = uuid.uuid4().hex[:12]
        now = _now()
        self.conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, title) VALUES (?, ?, ?, ?)",
            (session_id, now, now, title[:80]),
        )
        self.conn.commit()
        return session_id

    def latest_session_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS messages "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        now = _now()
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                message.get("role", ""),
                _searchable_text(message),
                json.dumps(message, ensure_ascii=False),
                now,
            ),
        )
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        self.conn.commit()

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT raw_json FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [json.loads(r["raw_json"]) for r in rows]

    def search_sessions(
        self, query: str, limit: int = 5, exclude_session: str | None = None
    ) -> list[SearchHit]:
        """Full-text search across all stored messages, best matches first."""
        expr = _fts_query(query)
        if not expr:
            return []
        sql = (
            "SELECT m.session_id, s.title, m.role, m.created_at, "
            "snippet(messages_fts, 0, '[', ']', ' ... ', 16) AS snippet "
            "FROM messages_fts "
            "JOIN messages m ON m.id = messages_fts.rowid "
            "JOIN sessions s ON s.id = m.session_id "
            "WHERE messages_fts MATCH ? "
        )
        params: list[Any] = [expr]
        if exclude_session:
            sql += "AND m.session_id != ? "
            params.append(exclude_session)
        sql += "ORDER BY rank LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        return [
            SearchHit(r["session_id"], r["title"], r["role"], r["snippet"], r["created_at"])
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
