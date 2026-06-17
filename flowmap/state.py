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
        """Run schema migrations from from_version to current.

        Only migrations that genuinely require re-embedding set needs_reindex —
        no-op version bumps must not nag the user to run an expensive re-index.
        """
        v = int(from_version)
        needs_reindex = False
        if v < 2:
            # v1 → v2: no structural changes, just a version bump.
            pass
        # Future migrations go here:
        # if v < 3:
        #     self._conn.execute("ALTER TABLE repos ADD COLUMN new_col TEXT")
        #     self._conn.commit()
        #     needs_reindex = True
        if needs_reindex:
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
        self._set_meta_conn(key, value, profile)
        self._conn.commit()

    def _set_meta_conn(self, key: str, value: str, profile: str = "default"):
        """Write one meta row WITHOUT committing — for batching into a transaction."""
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (profile, key, value) VALUES (?, ?, ?)",
            (profile, key, value),
        )

    # -- per-profile index state (profile-scoped via meta) -------------------

    def get_repo_index_sha(self, repo: str, profile: str = "default") -> str | None:
        """SHA the repo was last indexed at *for this profile*. None if never."""
        return self.get_meta(f"idx_sha:{repo}", profile)

    def get_repo_index_branch(self, repo: str, profile: str = "default") -> str | None:
        return self.get_meta(f"idx_branch:{repo}", profile)

    def get_repo_index(self, repo: str, profile: str = "default") -> dict:
        """Per-profile index record: {sha, branch, indexed_at}. `sha` is None if
        never indexed under this profile. For the legacy `default` profile only,
        falls back to the global `repos` table so pre-profile indexes still show.
        Single source of this fallback rule — callers must not re-implement it."""
        sha = self.get_repo_index_sha(repo, profile)
        branch = self.get_repo_index_branch(repo, profile)
        indexed_at = self.get_meta(f"idx_at:{repo}", profile)
        # Legacy `default` profile only: fall back to the global repos table for
        # pre-profile indexes. Never bleed the global (last-writer-wins) row into
        # a non-default profile's record.
        if sha is None and profile == "default":
            info = self.get_repo(repo)
            if info:
                sha = info.get("last_indexed_sha")
                branch = info.get("last_indexed_branch")
                if indexed_at is None:
                    indexed_at = info.get("last_indexed_at")
        return {"sha": sha, "branch": branch, "indexed_at": indexed_at}

    def set_repo_index(self, repo: str, sha: str, branch: str, chunks: int, profile: str = "default"):
        """Record that `repo` was indexed at `sha` under `profile`. Independent
        per profile so reindexing one model never marks another as fresh.

        All four fields are written in ONE transaction — a crash mid-write must
        not leave a torn record (sha set, timestamp missing) that reads as indexed."""
        with self._conn:  # commits on success, rolls back on exception
            self._set_meta_conn(f"idx_sha:{repo}", sha, profile)
            self._set_meta_conn(f"idx_branch:{repo}", branch or "", profile)
            self._set_meta_conn(f"idx_chunks:{repo}", str(chunks), profile)
            self._set_meta_conn(f"idx_at:{repo}", datetime.now(timezone.utc).isoformat(), profile)

    # Staleness keys for one repo, across every profile. Exact-match keys only —
    # global singletons (embedding_model/dims, schema_version) have no colon.
    _STALENESS_KEY_PREFIXES = ("idx_sha:", "idx_branch:", "idx_chunks:", "idx_at:", "pending:")

    def _clear_repo_staleness_conn(self, repo: str):
        """Delete this repo's staleness rows for all profiles. Caller owns the
        transaction so it can be atomic with a repos-row delete."""
        for prefix in self._STALENESS_KEY_PREFIXES:
            self._conn.execute("DELETE FROM meta WHERE key = ?", (f"{prefix}{repo}",))

    def clear_repo_staleness(self, repo: str):
        """Clear `repo`'s per-profile staleness across all profiles (own transaction)."""
        with self._conn:
            self._clear_repo_staleness_conn(repo)

    def clear_all_staleness(self):
        """Wipe all index metadata for every profile — staleness AND per-profile
        embedding_model/dims — backing `reset --all`. Only `schema_version` (a true
        global singleton) survives. GLOB so '_' is literal."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM meta WHERE key GLOB 'idx_sha:*' OR key GLOB 'idx_branch:*' "
                "OR key GLOB 'idx_chunks:*' OR key GLOB 'idx_at:*' OR key GLOB 'pending:*' "
                "OR key IN ('embedding_model', 'embedding_dims')"
            )

    def clear_profile_staleness(self, profile: str):
        """Drop one profile's staleness + per-profile model/dims meta (used when a
        profile's table is dropped, e.g. reset --benchmarks). Preserves schema_version."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM meta WHERE profile = ? AND ("
                "key GLOB 'idx_sha:*' OR key GLOB 'idx_branch:*' OR key GLOB 'idx_chunks:*' "
                "OR key GLOB 'idx_at:*' OR key GLOB 'pending:*' "
                "OR key IN ('embedding_model', 'embedding_dims'))",
                (profile,),
            )

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
        # Atomic: drop the repos row AND all per-profile staleness in one tx, so a
        # crash can't leave staleness pointing at vectors that were deleted.
        with self._conn:
            self._conn.execute("DELETE FROM repos WHERE name = ?", (name,))
            self._clear_repo_staleness_conn(name)
