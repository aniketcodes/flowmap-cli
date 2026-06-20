"""LanceDB vector store for FlowMap — profile-scoped tables, deterministic IDs."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa


log = logging.getLogger(__name__)

# Row-scan caps. Generous, but warn when hit rather than silently truncate (which
# would leave orphaned chunks or drop chunk mappings without any signal).
STALE_FILE_SCAN_CAP = 1_000_000   # distinct-file scan in delete_stale_files
STALE_CHUNK_SCAN_CAP = 100_000    # per-file chunk-id scan in delete_stale_chunks
FILE_CHUNKS_CAP = 10_000          # per-file chunk fetch in get_chunks_for_file
GET_STATS_SCAN_CAP = 100_000      # repo-column scan in get_stats fallback


class StoreError(Exception):
    """Raised when a store query fails unexpectedly.

    Callers should catch this and show a diagnostic message rather than
    letting users see 'No results found' when their index is broken.
    """

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _make_schema(vector_dims: int) -> pa.Schema:
    """Create PyArrow schema with a fixed-size vector column."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("repo", pa.string()),
        pa.field("file", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("extension", pa.string()),
        pa.field("language", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("symbol_name", pa.string()),
        pa.field("signature", pa.string()),
        pa.field("parent_symbol", pa.string()),
        pa.field("parent_signature", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=vector_dims)),
    ])


@dataclass
class SearchResult:
    """Store-agnostic search result — no LanceDB internals leak.

    score: higher is better (similarity, not distance).
    """
    repo: str
    file: str
    start_line: int
    end_line: int
    text: str
    symbol_name: str
    chunk_type: str
    signature: str
    parent_symbol: str
    parent_signature: str
    language: str
    score: float


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

def make_chunk_id(
    repo: str, file: str, symbol_name: str, chunk_type: str, chunk_index: int,
) -> str:
    """Content-based deterministic ID.

    - Symbol chunks: sha256(repo:file:symbol_name:chunk_index) — includes
      chunk_index as disambiguator for overloaded methods / same-named symbols.
    - Non-symbol chunks: sha256(repo:file:chunk_type:chunk_index)
    """
    if symbol_name:
        raw = f"{repo}:{file}:{symbol_name}:{chunk_index}"
    else:
        raw = f"{repo}:{file}:{chunk_type}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _escape_sql(value: str) -> str:
    """Escape special characters for LanceDB SQL equality predicates."""
    return value.replace("\\", "\\\\").replace("'", "''")


def _escape_like(value: str) -> str:
    """Escape for LIKE patterns — quote-safe AND wildcard-safe.

    Escapes single quotes (for SQL safety) plus % and _ (LIKE wildcards).
    Python-side filtering provides defense-in-depth.
    """
    return _escape_sql(value).replace("%", "\\%").replace("_", "\\_")


def _fts_score(row: dict) -> float:
    """BM25 score from an FTS row (`_score`, older builds `score`). Explicit
    None checks, not `a or b` — a real 0.0 score is falsy and must survive."""
    s = row.get("_score")
    if s is None:
        s = row.get("score")
    return float(s) if s is not None else 0.0


# FTS index schema version. Bump when rebuild_fts_index changes how the index is
# built so existing profiles get a one-time forced rebuild (see cli.index).
#   1 = legacy, no token positions
#   2 = with_position=True (enables quoted phrase queries)
FTS_INDEX_VERSION = "2"


class VectorStore:
    """LanceDB-backed vector store with profile-scoped tables."""

    def __init__(self, db_path: str | Path, vector_dims: int = 1024):
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))
        self._vector_dims = vector_dims
        self._table_names_cache: list[str] | None = None
        # Validated (profile, dims) pairs — keyed by dims too so a single store
        # instance touching tables of different dims can't poison the cache.
        self._dims_validated: set[tuple[str, int]] = set()

    def close(self):
        """Explicit cleanup."""
        pass  # LanceDB connections are lightweight; no explicit close needed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _list_table_names(self) -> list[str]:
        """Get table names with caching. Invalidated on create/drop."""
        if self._table_names_cache is not None:
            return self._table_names_cache
        result = self._db.list_tables()
        # list_tables() returns ListTablesResponse with .tables attribute
        if hasattr(result, "tables"):
            self._table_names_cache = result.tables
        else:
            self._table_names_cache = list(result)
        return self._table_names_cache

    def _invalidate_table_cache(self):
        self._table_names_cache = None

    def _validate_dims(self, table, profile: str = "default"):
        """Check table vector dims match configured dims. Raises StoreError on mismatch."""
        cache_key = (profile, self._vector_dims)
        if cache_key in self._dims_validated:
            return
        try:
            schema = table.schema
            vec_field = schema.field("vector")
            table_dims = vec_field.type.list_size
        except Exception as e:
            # Schema introspection failed — skip WITHOUT caching, so a later call
            # retries rather than silently trusting an unvalidated table forever.
            log.warning("Could not validate dims for profile %s: %s (skipping check)", profile, e)
            return
        if table_dims != self._vector_dims:
            raise StoreError(
                f"Dimension mismatch: index has {table_dims}-dim vectors but "
                f"current model produces {self._vector_dims}-dim. "
                f"Run: flowmap index --full"
            )
        self._dims_validated.add(cache_key)

    def _table_name(self, profile: str = "default") -> str:
        if profile == "default":
            return "code_index"
        return f"code_index__{profile}"

    def _get_or_create_table(self, profile: str = "default"):
        name = self._table_name(profile)
        if name in self._list_table_names():
            return self._db.open_table(name)
        table = self._db.create_table(name, schema=_make_schema(self._vector_dims))
        self._invalidate_table_cache()
        self._create_indexes(table)
        return table

    def _create_indexes(self, table):
        """Create vector, scalar, and FTS indexes. Safe to call multiple times.

        LanceDB raises generic Exception when an index already exists — no
        specific exception type is available, so we catch broadly here.
        """
        for col in ("repo", "symbol_name", "chunk_type"):
            try:
                table.create_scalar_index(col)
            except Exception as e:
                log.debug("Scalar index on %s already exists or failed: %s", col, e)

    def rebuild_fts_index(self, profile: str = "default") -> str | None:
        """Rebuild the FTS index. Call once after all upserts, not per-repo.

        Returns the schema version actually built, so the caller stamps what was
        achieved, not attempted:
          - "2" (FTS_INDEX_VERSION): positioned index built
          - "1": positionless fallback (older LanceDB)
          - None: nothing built (absent table or build failed)
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return None
        table = self._db.open_table(name)
        try:
            # with_position=True stores token positions so quoted phrase queries
            # ("connection pool") work instead of raising "position is not found".
            table.create_fts_index("text", replace=True, with_position=True)
            return FTS_INDEX_VERSION
        except TypeError:
            # Older LanceDB without with_position — fall back to a positionless
            # index and report "1" so the migration re-fires after a LanceDB upgrade.
            try:
                table.create_fts_index("text", replace=True)
                return "1"
            except Exception as e:
                log.debug("FTS index rebuild (positionless fallback) failed: %s", e)
                return None
        except Exception as e:
            log.debug("FTS index rebuild skipped: %s", e)
            return None

    def rebuild_vector_index(self, profile: str = "default"):
        """Rebuild the IVF-PQ vector index. Call after all upserts are done.

        Only creates the index if the table has enough rows (>= 5000) for IVF
        to be meaningful. Below that threshold, brute-force scan is fast enough.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return
        table = self._db.open_table(name)
        try:
            count = table.count_rows()
            if count < 5000:
                log.debug("Skipping vector index: %d rows (need >= 5000)", count)
                return
            # Scale partitions with table size: sqrt(n), clamped to [16, 512]
            num_partitions = max(16, min(512, int(count ** 0.5)))
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=min(96, self._vector_dims // 8),
                vector_column_name="vector",
                replace=True,
            )
            log.info("Vector index rebuilt: %d rows, %d partitions", count, num_partitions)
        except Exception as e:
            log.debug("Vector index rebuild skipped: %s", e)

    # -- write operations ----------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        profile: str = "default",
    ):
        """Insert or overwrite chunks by ID."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )

        table = self._get_or_create_table(profile)

        records = []
        for chunk, vector in zip(chunks, embeddings):
            records.append({
                "id": chunk["id"],
                "repo": chunk["repo"],
                "file": chunk["file"],
                "file_name": chunk["file_name"],
                "extension": chunk["extension"],
                "language": chunk.get("language", ""),
                "chunk_type": chunk.get("chunk_type", "fallback"),
                "symbol_name": chunk.get("symbol_name", ""),
                "signature": chunk.get("signature", ""),
                "parent_symbol": chunk.get("parent_symbol", ""),
                "parent_signature": chunk.get("parent_signature", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "vector": vector,
            })

        table.merge_insert("id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(records)

        # Note: FTS index rebuild moved to rebuild_fts_index() — call once after all upserts

    def delete_by_repo(self, repo: str, profile: str = "default"):
        """Delete all chunks for a repo."""
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return
        table = self._db.open_table(name)
        table.delete(f"repo = '{_escape_sql(repo)}'")

    def delete_by_file(self, repo: str, file: str, profile: str = "default"):
        """Delete all chunks for a specific file in a repo."""
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return
        table = self._db.open_table(name)
        table.delete(f"repo = '{_escape_sql(repo)}' AND file = '{_escape_sql(file)}'")

    def delete_stale_files(self, repo: str, current_files: set[str], profile: str = "default") -> bool:
        """Delete chunks for files no longer present in the repo.

        Used after upsert-first full reindex to clean up stale data without
        a delete-everything-first approach (which has a data-loss window).

        Returns True on success, False if cleanup failed (so the caller can keep
        the pending marker set and force a full reindex next run, rather than
        clearing it and leaving orphaned chunks unnoticed).
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return True

        try:
            table = self._db.open_table(name)
            # Get distinct files currently stored for this repo. A fixed limit would
            # silently miss stale files in very large repos (leaving orphaned chunks),
            # so cap generously and warn if we hit it rather than truncate silently.
            rows = table.search().select(["file"]).where(
                f"repo = '{_escape_sql(repo)}'"
            ).limit(STALE_FILE_SCAN_CAP).to_list()
            cap_hit = len(rows) >= STALE_FILE_SCAN_CAP
            if cap_hit:
                log.warning(
                    "delete_stale_files: repo %s has >= %d chunks; stale-file scan may be "
                    "incomplete and orphaned chunks could remain.", repo, STALE_FILE_SCAN_CAP
                )
            stored_files = {r["file"] for r in rows}

            stale_files = stored_files - current_files
            for f in stale_files:
                table.delete(f"repo = '{_escape_sql(repo)}' AND file = '{_escape_sql(f)}'")
            if stale_files:
                log.info("Deleted %d stale files for repo %s", len(stale_files), repo)
            # If we hit the scan cap, cleanup is known-incomplete — report failure so
            # the caller keeps the pending marker and forces a full reindex next run.
            return not cap_hit
        except Exception as e:
            log.warning("delete_stale_files failed for %s: %s (stale chunks may remain)", repo, e)
            return False

    def delete_stale_chunks(self, repo: str, file: str, valid_ids: set[str], profile: str = "default") -> bool:
        """Delete chunks for a specific file whose IDs are not in the valid set.

        Used after upsert-first incremental reindex: new chunks are upserted via
        merge_insert, then this cleans up stale IDs (removed symbols, reordered chunks).
        If crash occurs before this runs, stale chunks remain but no data is lost.

        Returns True on success, False if cleanup is known-incomplete (scan cap hit)
        or failed — so the caller keeps the pending marker and forces a full reindex
        next run rather than silently leaving orphaned chunks.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return True

        try:
            table = self._db.open_table(name)
            rows = table.search().select(["id"]).where(
                f"repo = '{_escape_sql(repo)}' AND file = '{_escape_sql(file)}'"
            ).limit(STALE_CHUNK_SCAN_CAP).to_list()
            cap_hit = len(rows) >= STALE_CHUNK_SCAN_CAP
            if cap_hit:
                log.warning(
                    "delete_stale_chunks: %s/%s has >= %d chunks; stale-chunk scan may be "
                    "incomplete and orphaned chunks could remain.", repo, file, STALE_CHUNK_SCAN_CAP
                )
            stale_ids = [r["id"] for r in rows if r["id"] not in valid_ids]
            for sid in stale_ids:
                table.delete(f"id = '{_escape_sql(sid)}'")
            if stale_ids:
                log.debug("Deleted %d stale chunks for %s/%s", len(stale_ids), repo, file)
            return not cap_hit
        except Exception as e:
            log.warning("delete_stale_chunks failed for %s/%s: %s", repo, file, e)
            return False

    # -- search operations ---------------------------------------------------

    def search_vector(
        self,
        query_vector: list[float],
        limit: int = 10,
        repo_filter: str | None = None,
        profile: str = "default",
    ) -> list[SearchResult]:
        """Cosine similarity search. Returns results with score: higher = better.

        Raises StoreError on query failure or dimension mismatch.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return []

        try:
            table = self._db.open_table(name)
            self._validate_dims(table, profile)
            query = table.search(query_vector, vector_column_name="vector").limit(limit)

            if repo_filter:
                query = query.where(f"repo = '{_escape_sql(repo_filter)}'")

            results = query.to_list()
            return [self._row_to_result(r) for r in results]
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"Vector search failed: {e}") from e

    def search_fts(
        self,
        query: str,
        limit: int = 10,
        repo_filter: str | None = None,
        profile: str = "default",
    ) -> list[SearchResult]:
        """Full-text BM25 search via the LanceDB FTS index on the `text` column.

        Returns results scored by BM25 relevance (higher = better). Best-effort,
        like the ripgrep leg: degrades to an empty list if the table or its FTS
        index is absent, or the query is rejected by the FTS parser.
        """
        if not query or not query.strip():
            return []
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return []
        table = self._db.open_table(name)
        # Best-effort: does this table actually carry an FTS index on `text`?
        # Lets us tell "not built yet" (debug) from "built but broken" (warning),
        # so a silently-degraded BM25 leg is visible rather than indistinguishable
        # from an un-rebuilt profile. Wrapped — detection must never break search.
        has_fts: bool | None = None
        try:
            has_fts = any(
                "text" in (getattr(ix, "columns", None) or [])
                and "fts" in str(getattr(ix, "index_type", "")).lower()
                for ix in table.list_indices()
            )
        except Exception:
            pass
        try:
            q = table.search(query, query_type="fts").limit(limit)
            if repo_filter:
                q = q.where(f"repo = '{_escape_sql(repo_filter)}'")
            out: list[SearchResult] = []
            for r in q.to_list():
                sr = self._row_to_result(r)
                sr.score = _fts_score(r)
                out.append(sr)
            return out
        except Exception as e:
            # This leg is optional, so degrade gracefully rather than fail search.
            if has_fts:
                log.warning("FTS search failed on existing index for %s: %s", name, e)
            else:
                log.debug("FTS search unavailable (index absent?) for %s: %s", name, e)
            return []

    def search_symbol(
        self,
        symbol_name: str,
        repo_filter: str | None = None,
        file_filter: str | None = None,
        limit: int = 10,
        profile: str = "default",
    ) -> list[SearchResult]:
        """Search by symbol name — exact match, suffix match (.Name), and contains match.

        If file_filter is provided, results are scoped to that file path only.
        Raises StoreError on query failure.

        LIKE patterns use _escape_sql (quote-safe only). Underscore and percent
        act as SQL wildcards in the LIKE — Python-side filtering ensures correctness.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return []

        try:
            table = self._db.open_table(name)
            safe_sym = _escape_sql(symbol_name)
            repo_clause = f" AND repo = '{_escape_sql(repo_filter)}'" if repo_filter else ""
            file_clause = f" AND file = '{_escape_sql(file_filter)}'" if file_filter else ""

            results: list[SearchResult] = []

            # 1. Exact match (score 1.0) — uses = not LIKE, no wildcard issue
            rows = table.search().where(f"symbol_name = '{safe_sym}'{repo_clause}{file_clause}").limit(limit).to_list()
            for r in rows:
                sr = self._row_to_result(r)
                sr.score = 1.0
                results.append(sr)

            # 2. Suffix match — "Process" matches "Service.Process" (score 0.8)
            # LIKE fetches candidates broadly; Python filter ensures exact suffix
            if len(results) < limit:
                safe_sym_like = _escape_like(symbol_name)
                suffix_where = f"symbol_name LIKE '%.{safe_sym_like}' ESCAPE '\\'{repo_clause}{file_clause}"
                rows = table.search().where(suffix_where).limit(limit * 3).to_list()
                seen = {(r.repo, r.file, r.start_line) for r in results}
                for r in rows:
                    if not r.get("symbol_name", "").endswith(f".{symbol_name}"):
                        continue
                    sr = self._row_to_result(r)
                    key = (sr.repo, sr.file, sr.start_line)
                    if key not in seen:
                        sr.score = 0.8
                        results.append(sr)
                        seen.add(key)

            # 3. Contains match — "validate" matches "AuthService.validate_token" (score 0.5)
            # LIKE fetches candidates broadly; Python filter ensures exact containment
            if len(results) < limit:
                safe_sym_like = _escape_like(symbol_name)
                contains_where = f"symbol_name LIKE '%{safe_sym_like}%' ESCAPE '\\'{repo_clause}{file_clause}"
                rows = table.search().where(contains_where).limit(limit * 3).to_list()
                seen = {(r.repo, r.file, r.start_line) for r in results}
                for r in rows:
                    if symbol_name not in r.get("symbol_name", ""):
                        continue
                    sr = self._row_to_result(r)
                    key = (sr.repo, sr.file, sr.start_line)
                    if key not in seen:
                        sr.score = 0.5
                        results.append(sr)
                        seen.add(key)

            results.sort(key=lambda r: r.score, reverse=True)
            return results
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"Symbol search failed: {e}") from e

    def find_chunk_containing(
        self,
        repo: str,
        file: str,
        line: int,
        profile: str = "default",
    ) -> SearchResult | None:
        """Find the chunk that contains a given line number.

        Used by hybrid search to map ripgrep line hits to stored chunks
        for dedup-before-scoring.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return None

        table = self._db.open_table(name)
        where = (
            f"repo = '{_escape_sql(repo)}' AND "
            f"file = '{_escape_sql(file)}' AND "
            f"start_line <= {int(line)} AND "
            f"end_line >= {int(line)}"
        )

        try:
            rows = table.search().where(where).limit(10).to_list()
        except Exception as e:
            raise StoreError(f"find_chunk_containing failed: {e}") from e

        if not rows:
            return None

        # Prefer the most specific (smallest span) chunk when overlapping
        rows.sort(key=lambda r: r.get("end_line", 0) - r.get("start_line", 0))
        return self._row_to_result(rows[0])

    def get_chunks_for_file(
        self,
        repo: str,
        file: str,
        profile: str = "default",
    ) -> list[SearchResult]:
        """Get all chunks for a specific file. Returns chunks sorted by start_line.

        Used by hybrid search for batch ripgrep-to-chunk mapping — one query per
        unique file instead of one per ripgrep hit.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return []

        table = self._db.open_table(name)
        where = f"repo = '{_escape_sql(repo)}' AND file = '{_escape_sql(file)}'"

        try:
            rows = table.search().where(where).limit(FILE_CHUNKS_CAP).to_list()
            if len(rows) >= FILE_CHUNKS_CAP:
                log.warning(
                    "get_chunks_for_file: %s/%s has >= %d chunks; some ripgrep hits may not "
                    "map to a chunk and fusion may be degraded.", repo, file, FILE_CHUNKS_CAP
                )
            chunks = [self._row_to_result(r) for r in rows]
            chunks.sort(key=lambda c: c.start_line)
            return chunks
        except Exception as e:
            raise StoreError(f"get_chunks_for_file failed for {repo}/{file}: {e}") from e

    # -- stats ---------------------------------------------------------------

    def get_stats(self, profile: str = "default", known_repos: list[str] | None = None) -> dict:
        """Return chunk counts grouped by repo. Uses per-repo count_rows to avoid loading full table."""
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return {"total": 0, "repos": {}}

        table = self._db.open_table(name)
        total = table.count_rows()
        if total == 0:
            return {"total": 0, "repos": {}}

        repo_counts: dict[str, int] = {}
        if known_repos:
            for repo in known_repos:
                try:
                    count = table.count_rows(f"repo = '{_escape_sql(repo)}'")
                    if count > 0:
                        repo_counts[repo] = count
                except Exception as e:
                    log.debug("get_stats count_rows failed for %s: %s", repo, e)
        else:
            # Fallback: read just repo column (lighter than full table)
            try:
                rows = table.search().select(["repo"]).limit(min(total, GET_STATS_SCAN_CAP)).to_list()
                if total > GET_STATS_SCAN_CAP:
                    log.warning(
                        "get_stats: table %s has %d chunks (> %d cap); per-repo counts "
                        "are truncated and may under-report.", name, total, GET_STATS_SCAN_CAP
                    )
                from collections import Counter
                repo_counts = dict(Counter(r["repo"] for r in rows))
            except Exception as e:
                log.debug("get_stats repo enumeration failed: %s", e)

        return {"total": total, "repos": repo_counts}

    # -- map & symbols (metadata queries, no vectors) -------------------------

    def get_repo_map(self, repo: str | None = None, profile: str = "default") -> tuple[list[dict], bool]:
        """Get structural metadata for map command.

        Returns (rows, truncated). Raises StoreError on query failure.
        """
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return [], False

        try:
            table = self._db.open_table(name)
            select_cols = ["repo", "file", "symbol_name", "chunk_type", "signature",
                           "start_line", "end_line", "language", "parent_symbol"]

            where = f"repo = '{_escape_sql(repo)}'" if repo else None

            query = table.search().select(select_cols)
            if where:
                query = query.where(where)
            rows = query.limit(50000).to_list()
            truncated = len(rows) >= 50000
            if truncated:
                log.warning("get_repo_map: results truncated at 50,000 rows")
            return rows, truncated
        except Exception as e:
            raise StoreError(f"get_repo_map failed: {e}") from e

    def get_symbols(
        self,
        query: str | None = None,
        repo: str | None = None,
        kind: str | None = None,
        limit: int = 100,
        profile: str = "default",
    ) -> list[dict]:
        """Get symbols for symbols command. Filters by name (contains), repo, and chunk_type."""
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return []

        table = self._db.open_table(name)
        select_cols = ["repo", "file", "symbol_name", "chunk_type", "signature",
                       "start_line", "end_line", "language", "parent_symbol"]

        conditions = ["symbol_name != ''"]  # exclude preamble/fallback chunks
        if repo:
            conditions.append(f"repo = '{_escape_sql(repo)}'")
        if kind:
            conditions.append(f"chunk_type = '{_escape_sql(kind)}'")

        # For LIKE, use _escape_sql (quote-safe) and filter in Python for correctness
        has_query_filter = bool(query)
        if query:
            conditions.append(f"symbol_name LIKE '%{_escape_like(query)}%' ESCAPE '\\'")

        where = " AND ".join(conditions)

        try:
            rows = table.search().select(select_cols).where(where).limit(limit * 3 if has_query_filter else limit).to_list()
            # Python-side filter ensures exact containment (LIKE _ wildcard won't cause false matches)
            if has_query_filter:
                rows = [r for r in rows if query in r.get("symbol_name", "")][:limit]
            return rows
        except Exception as e:
            raise StoreError(f"get_symbols failed: {e}") from e

    # -- table management ----------------------------------------------------

    def compact(self, profile: str = "default"):
        """Compact the table to reclaim space from deleted rows."""
        name = self._table_name(profile)
        if name not in self._list_table_names():
            return
        try:
            table = self._db.open_table(name)
            table.optimize()
            log.info("Compacted table %s", name)
        except Exception as e:
            log.debug("Compact skipped: %s", e)

    def drop_table(self, profile: str = "default"):
        """Drop a table entirely."""
        name = self._table_name(profile)
        if name in self._list_table_names():
            self._db.drop_table(name)
            self._invalidate_table_cache()

    def list_profiles(self) -> list[str]:
        """List all existing profiles."""
        profiles = []
        for name in self._list_table_names():
            if name == "code_index":
                profiles.append("default")
            elif name.startswith("code_index__"):
                profiles.append(name.removeprefix("code_index__"))
        return profiles

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_result(row: dict) -> SearchResult:
        # LanceDB _distance: lower = more similar (cosine distance)
        # Convert to score: higher = better
        distance = row.get("_distance", 0.0)
        if distance is None:
            distance = 0.0
        score = 1.0 / (1.0 + distance)

        return SearchResult(
            repo=row.get("repo", ""),
            file=row.get("file", ""),
            start_line=row.get("start_line", 0),
            end_line=row.get("end_line", 0),
            text=row.get("text", ""),
            symbol_name=row.get("symbol_name", ""),
            chunk_type=row.get("chunk_type", ""),
            signature=row.get("signature", ""),
            parent_symbol=row.get("parent_symbol", ""),
            parent_signature=row.get("parent_signature", ""),
            language=row.get("language", ""),
            score=score,
        )
