"""Integration tests for VectorStore — real LanceDB, deterministic embeddings."""

import pytest

from flowmap.store import StoreError, VectorStore, make_chunk_id
from tests.conftest import DIMS, hash_vector


def test_get_chunks_for_file_warns_when_cap_hit(tmp_path, monkeypatch, caplog):
    """Hitting the per-file chunk cap must warn, not silently drop chunk mappings."""
    import logging
    import flowmap.store as store_mod
    monkeypatch.setattr(store_mod, "FILE_CHUNKS_CAP", 2)
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    chunks, embs = _make_chunks("r", "a.py", ["f1", "f2", "f3"])  # 3 > cap of 2
    s.upsert_chunks(chunks, embs)
    with caplog.at_level(logging.WARNING):
        s.get_chunks_for_file("r", "a.py")
    assert any("get_chunks_for_file" in r.message and ">=" in r.message for r in caplog.records)
    s.close()


def test_delete_stale_chunks_returns_bool(tmp_path, monkeypatch):
    """Incremental cleanup must SIGNAL failure (like delete_stale_files) so the
    caller keeps the pending marker instead of clearing it over orphaned chunks."""
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    chunks, embs = _make_chunks("r", "a.py", ["f1", "f2"])
    s.upsert_chunks(chunks, embs)

    # Success → True
    assert s.delete_stale_chunks("r", "a.py", valid_ids={chunks[0]["id"], chunks[1]["id"]}) is True

    # Cap hit → False (known-incomplete scan)
    import flowmap.store as store_mod
    monkeypatch.setattr(store_mod, "STALE_CHUNK_SCAN_CAP", 1)
    assert s.delete_stale_chunks("r", "a.py", valid_ids=set()) is False
    monkeypatch.undo()

    # Exception during delete → False
    real_open = s._db.open_table
    def boom(name):
        t = real_open(name)
        monkeypatch.setattr(t, "delete", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
        return t
    monkeypatch.setattr(s._db, "open_table", boom)
    assert s.delete_stale_chunks("r", "a.py", valid_ids=set()) is False
    s.close()


def test_delete_stale_files_returns_false_when_cap_hit(tmp_path, monkeypatch):
    """Hitting the scan cap means cleanup is known-incomplete → return False so the
    caller keeps the pending marker (rather than silently leaving orphans)."""
    import flowmap.store as store_mod
    monkeypatch.setattr(store_mod, "STALE_FILE_SCAN_CAP", 1)
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    chunks, embs = _make_chunks("r", "a.py", ["f1", "f2"])  # 2 rows >= cap of 1
    s.upsert_chunks(chunks, embs)
    assert s.delete_stale_files("r", {"a.py"}, profile="default") is False
    s.close()


def test_get_stats_fallback_warns_when_truncated(tmp_path, monkeypatch, caplog):
    import logging
    import flowmap.store as store_mod
    monkeypatch.setattr(store_mod, "GET_STATS_SCAN_CAP", 1)
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    chunks, embs = _make_chunks("r", "a.py", ["f1", "f2"])  # total 2 > cap 1
    s.upsert_chunks(chunks, embs)
    with caplog.at_level(logging.WARNING):
        s.get_stats()  # no known_repos → fallback path
    assert any("get_stats" in r.message and "truncated" in r.message for r in caplog.records)
    s.close()


def test_delete_stale_files_returns_false_on_failure(tmp_path, monkeypatch):
    """Cleanup failure must be signalled (False), not swallowed silently — so the
    caller can keep the pending marker and force a full reindex next run."""
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    chunks, embs = _make_chunks("r", "a.py", ["f1"])
    s.upsert_chunks(chunks, embs)

    real_open = s._db.open_table
    def boom(name):
        t = real_open(name)
        monkeypatch.setattr(t, "delete", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
        return t
    monkeypatch.setattr(s._db, "open_table", boom)

    # current_files omits a.py → it's stale → delete attempted → raises → False
    assert s.delete_stale_files("r", set(), profile="default") is False
    s.close()


def test_search_raises_on_dimension_mismatch(tmp_path):
    """A table built at one dim, queried by a store configured for another dim,
    must raise StoreError — the automated guard for switching profiles whose
    embedding models have different vector sizes (e.g. qwen 0.6b=1024 vs 4b=2560)."""
    path = tmp_path / "lancedb"
    chunk = {
        "id": make_chunk_id(repo="r", file="a.py", symbol_name="f", chunk_type="function", chunk_index=0),
        "repo": "r", "file": "a.py", "file_name": "a.py", "extension": ".py",
        "language": "python", "chunk_type": "function", "symbol_name": "f",
        "signature": "def f():", "parent_symbol": "", "parent_signature": "",
        "start_line": 1, "end_line": 3, "chunk_index": 0, "text": "def f(): pass",
    }
    with VectorStore(path, vector_dims=8) as s:
        s.upsert_chunks([chunk], [hash_vector("r:a.py:f", 8)])
    # Reopen the same table with a different configured dim → mismatch on query.
    with VectorStore(path, vector_dims=16) as s:
        with pytest.raises(StoreError):
            s.search_vector(hash_vector("r:a.py:f", 16), limit=5)


def _make_chunks(repo: str, file: str, symbols: list[str]) -> tuple[list[dict], list[list[float]]]:
    chunks = []
    embeddings = []
    for i, sym in enumerate(symbols):
        cid = make_chunk_id(repo=repo, file=file, symbol_name=sym, chunk_type="function", chunk_index=i)
        chunks.append({
            "id": cid, "repo": repo, "file": file, "file_name": file.split("/")[-1],
            "extension": ".py", "language": "python", "chunk_type": "function",
            "symbol_name": sym, "signature": f"def {sym}():", "parent_symbol": "",
            "parent_signature": "", "start_line": i * 10 + 1, "end_line": (i + 1) * 10,
            "chunk_index": i, "text": f"def {sym}():\n    pass",
        })
        embeddings.append(hash_vector(f"{repo}:{file}:{sym}"))
    return chunks, embeddings


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "lancedb", vector_dims=DIMS)
    yield s
    s.close()


class TestUpsertAndSearch:
    def test_upsert_and_search_vector(self, store):
        chunks, embeddings = _make_chunks("repo1", "src/main.py", ["hello", "world"])
        store.upsert_chunks(chunks, embeddings)

        query_vec = hash_vector("repo1:src/main.py:hello")
        results = store.search_vector(query_vec, limit=5)
        assert len(results) >= 1
        assert results[0].symbol_name == "hello"

    def test_upsert_and_search_symbol_exact(self, store):
        chunks, embeddings = _make_chunks("repo1", "src/main.py", ["process_data", "validate"])
        store.upsert_chunks(chunks, embeddings)

        results = store.search_symbol("process_data", repo_filter="repo1")
        assert len(results) >= 1
        assert results[0].symbol_name == "process_data"
        assert results[0].score == 1.0

    def test_search_symbol_contains(self, store):
        chunks, embeddings = _make_chunks("repo1", "src/main.py", ["validate_token", "parse_input"])
        store.upsert_chunks(chunks, embeddings)

        results = store.search_symbol("validate", repo_filter="repo1")
        assert any(r.symbol_name == "validate_token" for r in results)

    def test_search_symbol_excludes_other_repo(self, store):
        c1, e1 = _make_chunks("repo1", "a.py", ["foo"])
        c2, e2 = _make_chunks("repo2", "b.py", ["bar"])
        store.upsert_chunks(c1, e1)
        store.upsert_chunks(c2, e2)

        results = store.search_symbol("foo", repo_filter="repo2")
        assert not any(r.symbol_name == "foo" for r in results)

    def test_search_vector_with_repo_filter(self, store):
        c1, e1 = _make_chunks("repo1", "a.py", ["foo"])
        c2, e2 = _make_chunks("repo2", "b.py", ["bar"])
        store.upsert_chunks(c1, e1)
        store.upsert_chunks(c2, e2)

        query_vec = hash_vector("repo1:a.py:foo")
        results = store.search_vector(query_vec, limit=5, repo_filter="repo1")
        assert all(r.repo == "repo1" for r in results)


class TestDeleteOperations:
    def test_delete_by_repo(self, store):
        c1, e1 = _make_chunks("repo1", "a.py", ["foo"])
        c2, e2 = _make_chunks("repo2", "b.py", ["bar"])
        store.upsert_chunks(c1, e1)
        store.upsert_chunks(c2, e2)

        store.delete_by_repo("repo1")
        results = store.search_symbol("foo", repo_filter="repo1")
        assert len(results) == 0
        # repo2 still there
        results2 = store.search_symbol("bar", repo_filter="repo2")
        assert len(results2) >= 1

    def test_delete_stale_files(self, store):
        """delete_stale_files removes chunks for files no longer in the current set."""
        c1, e1 = _make_chunks("repo1", "a.py", ["foo"])
        c2, e2 = _make_chunks("repo1", "b.py", ["bar"])
        c3, e3 = _make_chunks("repo1", "c.py", ["baz"])
        store.upsert_chunks(c1, e1)
        store.upsert_chunks(c2, e2)
        store.upsert_chunks(c3, e3)

        # Simulate re-index where c.py was deleted from the repo
        store.delete_stale_files("repo1", current_files={"a.py", "b.py"})

        # a.py and b.py still there
        results = store.search_symbol("foo", repo_filter="repo1")
        assert len(results) >= 1
        results = store.search_symbol("bar", repo_filter="repo1")
        assert len(results) >= 1
        # c.py is gone
        results = store.search_symbol("baz", repo_filter="repo1")
        assert len(results) == 0

    def test_delete_by_file(self, store):
        c1, e1 = _make_chunks("repo1", "a.py", ["foo"])
        c2, e2 = _make_chunks("repo1", "b.py", ["bar"])
        store.upsert_chunks(c1, e1)
        store.upsert_chunks(c2, e2)

        store.delete_by_file("repo1", "a.py")
        results = store.search_symbol("foo", repo_filter="repo1")
        assert len(results) == 0
        results2 = store.search_symbol("bar", repo_filter="repo1")
        assert len(results2) >= 1


class TestMetadataQueries:
    def test_get_stats(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo", "bar"])
        store.upsert_chunks(chunks, embeddings)
        stats = store.get_stats(known_repos=["repo1"])
        assert stats["total"] == 2
        assert stats["repos"]["repo1"] == 2

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total"] == 0

    def test_get_repo_map(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["MyClass", "helper_fn"])
        store.upsert_chunks(chunks, embeddings)
        rows, truncated = store.get_repo_map(repo="repo1")
        assert not truncated
        assert len(rows) >= 2
        symbols = {r["symbol_name"] for r in rows}
        assert "MyClass" in symbols
        assert "helper_fn" in symbols

    def test_get_symbols(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["process", "validate"])
        store.upsert_chunks(chunks, embeddings)
        rows = store.get_symbols(query="proc", repo="repo1")
        assert len(rows) >= 1
        assert rows[0]["symbol_name"] == "process"

    def test_get_chunks_for_file(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo", "bar", "baz"])
        store.upsert_chunks(chunks, embeddings)
        file_chunks = store.get_chunks_for_file("repo1", "a.py")
        assert len(file_chunks) == 3
        # Should be sorted by start_line
        assert file_chunks[0].start_line <= file_chunks[1].start_line


class TestUpsertValidation:
    def test_upsert_mismatched_lengths_raises(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo", "bar"])
        # Remove one embedding to create mismatch
        with pytest.raises(ValueError, match="must have equal length"):
            store.upsert_chunks(chunks, embeddings[:1])


class TestDuplicateHandling:
    def test_upsert_same_id_updates(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)

        # Modify text and re-upsert with same ID
        chunks[0]["text"] = "def foo():\n    return 42"
        store.upsert_chunks(chunks, embeddings)

        stats = store.get_stats(known_repos=["repo1"])
        assert stats["total"] == 1  # not 2


class TestIndexes:
    def test_rebuild_fts_index(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)
        # Should not raise
        store.rebuild_fts_index()

    def test_rebuild_vector_index_skips_small(self, store):
        """Vector index requires >= 5000 rows. Small tables should skip without error."""
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)
        store.rebuild_vector_index()  # should not raise


class TestTableManagement:
    def test_drop_table(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)
        store.drop_table()
        stats = store.get_stats()
        assert stats["total"] == 0

    def test_list_profiles(self, store):
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings, profile="default")
        profiles = store.list_profiles()
        assert "default" in profiles


class TestSqlEscaping:
    """Verify SQL escaping prevents injection and handles special characters."""

    def test_single_quote_in_repo_name(self, store):
        """Repo names with single quotes don't cause SQL injection."""
        chunks, embeddings = _make_chunks("repo'inject", "a.py", ["safe_func"])
        store.upsert_chunks(chunks, embeddings)
        results = store.search_symbol("safe_func", repo_filter="repo'inject")
        assert len(results) >= 1
        assert results[0].repo == "repo'inject"

    def test_underscore_in_symbol_name(self, store):
        """Symbol names with underscores are found by exact match."""
        chunks, embeddings = _make_chunks("repo1", "a.py", ["__init__", "process_data"])
        store.upsert_chunks(chunks, embeddings)
        results = store.search_symbol("__init__", repo_filter="repo1")
        assert any(r.symbol_name == "__init__" for r in results)

    def test_delete_with_special_chars(self, store):
        """Delete operations handle special characters in repo/file names."""
        chunks, embeddings = _make_chunks("repo'test", "file'name.py", ["func"])
        store.upsert_chunks(chunks, embeddings)
        store.delete_by_repo("repo'test")
        results = store.search_symbol("func", repo_filter="repo'test")
        assert len(results) == 0


class TestStoreErrors:
    """Verify store errors propagate instead of being swallowed."""

    def test_search_vector_wrong_dims_raises(self, store):
        """Querying with wrong vector dims raises StoreError."""
        from flowmap.store import StoreError
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)
        wrong_dims_vector = [0.1] * 64  # store has 32-dim vectors
        with pytest.raises(StoreError):
            store.search_vector(wrong_dims_vector)

    def test_search_after_drop_raises(self, store):
        """StoreError propagates when table is missing (not silently empty)."""
        from flowmap.store import StoreError
        chunks, embeddings = _make_chunks("repo1", "a.py", ["foo"])
        store.upsert_chunks(chunks, embeddings)
        store.drop_table()
        # search_symbol on missing table returns [] (table not found is not an error)
        results = store.search_symbol("foo")
        assert results == []
        # But search_vector on missing table also returns []
        results = store.search_vector([0.1] * DIMS)
        assert results == []


class TestSymbolSearchAccuracy:
    """Verify LIKE patterns with Python-side filtering produce correct results."""

    def test_underscore_not_wildcard_in_contains(self, store):
        """Searching for __init__ via contains does not match 'xinita'."""
        chunks1, e1 = _make_chunks("repo1", "a.py", ["__init__"])
        chunks2, e2 = _make_chunks("repo1", "b.py", ["xinita"])
        store.upsert_chunks(chunks1, e1)
        store.upsert_chunks(chunks2, e2)
        results = store.search_symbol("__init__", repo_filter="repo1")
        names = [r.symbol_name for r in results]
        assert "__init__" in names
        assert "xinita" not in names

    def test_suffix_match_accuracy(self, store):
        """Suffix match for 'process' finds 'Service.process' but not 'preprocessor'."""
        chunks, embeddings = _make_chunks("repo1", "a.py", ["Service.process", "preprocessor", "process"])
        store.upsert_chunks(chunks, embeddings)
        results = store.search_symbol("process", repo_filter="repo1")
        names = [r.symbol_name for r in results]
        # Exact match
        assert "process" in names
        # Suffix match — Service.process ends with .process
        assert "Service.process" in names

    def test_percent_in_symbol_name(self, store):
        """Symbol with % is found via exact match, LIKE wildcard doesn't expand."""
        chunks, embeddings = _make_chunks("repo1", "a.py", ["calc_100%"])
        store.upsert_chunks(chunks, embeddings)
        results = store.search_symbol("calc_100%", repo_filter="repo1")
        names = [r.symbol_name for r in results]
        assert "calc_100%" in names
