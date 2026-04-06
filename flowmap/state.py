"""SQLite state tracking for FlowMap — repos, files, metadata (profile-aware)."""

from __future__ import annotations

import sqlite3
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_SCHEMA_VERSION = "2"

_INIT_SQL = """\
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS repos (
    name            TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    last_indexed_sha    TEXT,
    last_indexed_branch TEXT,
    last_indexed_at     TEXT,
    chunk_count         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    profile TEXT NOT NULL DEFAULT 'default',
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (profile, key)
);
"""


class StateDB:
    """Thin wrapper around SQLite for flowmap state."""

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_INIT_SQL)
        self._ensure_schema_version()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- schema versioning ---------------------------------------------------

    def _ensure_schema_version(self):
        stored = self.get_meta("schema_version")
        if stored is None:
            self.set_meta("schema_version", _SCHEMA_VERSION)
        elif stored != _SCHEMA_VERSION:
            self._migrate(stored)
            self.set_meta("schema_version", _SCHEMA_VERSION)

    def _migrate(self, from_version: str):
        """Run schema migrations from from_version to current."""
        v = int(from_version)
        if v < 2:
            # v1 → v2: no structural changes, just version bump
            pass
        # Future migrations go here:
        # if v < 3:
        #     self._conn.execute("ALTER TABLE repos ADD COLUMN new_col TEXT")
        #     self._conn.commit()
        warnings.warn(
            f"FlowMap DB migrated from schema v{from_version} to v{_SCHEMA_VERSION}. "
            "A full re-index is recommended: flowmap index --full",
            stacklevel=3,
        )

    # -- transaction helper --------------------------------------------------

    @contextmanager
    def transaction(self):
        """Batch multiple writes in a single transaction."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- meta (profile-scoped) -----------------------------------------------

    def get_meta(self, key: str, profile: str = "default") -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE profile = ? AND key = ?",
            (profile, key),
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str, profile: str = "default"):
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (profile, key, value) VALUES (?, ?, ?)",
            (profile, key, value),
        )
        self._conn.commit()

    # -- repos ----------------------------------------------------------------

    def upsert_repo(self, name: str, path: str):
        """Insert or update repo path. Does NOT reset indexing metadata."""
        self._conn.execute(
            "INSERT INTO repos (name, path) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET path = excluded.path",
            (name, path),
        )
        self._conn.commit()

    def get_repo(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM repos WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_repos(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM repos ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def update_repo_indexed(
        self, name: str, sha: str, branch: str, chunk_count: int
    ):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE repos SET last_indexed_sha = ?, last_indexed_branch = ?, "
            "last_indexed_at = ?, chunk_count = ? WHERE name = ?",
            (sha, branch, now, chunk_count, name),
        )
        self._conn.commit()

    def delete_repo(self, name: str):
        with self._conn:
            self._conn.execute("DELETE FROM repos WHERE name = ?", (name,))
