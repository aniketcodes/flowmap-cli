"""Tests for StateDB — SQLite state tracking."""

import warnings

import pytest

from flowmap.state import StateDB


@pytest.fixture
def state(tmp_path):
    db = StateDB(tmp_path / "test.db")
    yield db
    db.close()


class TestMetaOperations:
    def test_get_set_meta(self, state):
        state.set_meta("key1", "value1")
        assert state.get_meta("key1") == "value1"

    def test_get_meta_missing(self, state):
        assert state.get_meta("nonexistent") is None

    def test_meta_profile_scoping(self, state):
        state.set_meta("key1", "val_default")
        state.set_meta("key1", "val_bench", profile="bench")
        assert state.get_meta("key1") == "val_default"
        assert state.get_meta("key1", profile="bench") == "val_bench"

    def test_meta_overwrite(self, state):
        state.set_meta("key1", "v1")
        state.set_meta("key1", "v2")
        assert state.get_meta("key1") == "v2"


class TestPerProfileIndexState:
    """Indexed SHA/branch/chunks are tracked per profile, since the same repo
    can be indexed under multiple embedding models independently."""

    def test_set_and_get_per_profile(self, state):
        state.set_repo_index("repo-a", sha="sha06", branch="main", chunks=10, profile="qwen06b")
        state.set_repo_index("repo-a", sha="sha4b", branch="main", chunks=12, profile="qwen4b")

        assert state.get_repo_index_sha("repo-a", "qwen06b") == "sha06"
        assert state.get_repo_index_sha("repo-a", "qwen4b") == "sha4b"
        assert state.get_repo_index_branch("repo-a", "qwen06b") == "main"

    def test_missing_returns_none(self, state):
        assert state.get_repo_index_sha("repo-a", "qwen06b") is None
        assert state.get_repo_index_branch("repo-a", "qwen06b") is None

    def test_reindex_one_profile_does_not_touch_other(self, state):
        state.set_repo_index("repo-a", sha="old", branch="main", chunks=5, profile="qwen06b")
        state.set_repo_index("repo-a", sha="old", branch="main", chunks=5, profile="qwen4b")
        # Reindex only qwen06b at a new SHA
        state.set_repo_index("repo-a", sha="new", branch="main", chunks=7, profile="qwen06b")
        assert state.get_repo_index_sha("repo-a", "qwen06b") == "new"
        assert state.get_repo_index_sha("repo-a", "qwen4b") == "old"  # untouched / now stale


class TestSetRepoIndexAtomic:
    def test_partial_write_rolls_back(self, state, monkeypatch):
        """A failure mid set_repo_index must leave NO partial record — otherwise a
        torn write (sha set, idx_at missing) reads as 'indexed' and the next run
        wrongly skips it as up to date."""
        orig = state._set_meta_conn
        def boom(key, value, profile):
            if key.startswith("idx_at:"):
                raise RuntimeError("crash mid-write")
            return orig(key, value, profile)
        monkeypatch.setattr(state, "_set_meta_conn", boom)

        with pytest.raises(RuntimeError):
            state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")

        # All four keys rolled back — not a torn half-record.
        assert state.get_repo_index_sha("repo-a", "p1") is None
        assert state.get_repo_index_branch("repo-a", "p1") is None
        assert state.get_meta("idx_chunks:repo-a", "p1") is None

    def test_all_fields_written_on_success(self, state):
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        rec = state.get_repo_index("repo-a", "p1")
        assert rec["sha"] == "s1" and rec["branch"] == "main" and rec["indexed_at"]


class TestPerProfileIndexedAt:
    def test_indexed_at_is_per_profile(self, state):
        state.upsert_repo("repo-a", "/tmp/repo-a")
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        state.set_repo_index("repo-a", sha="s2", branch="main", chunks=6, profile="p2")
        at1 = state.get_repo_index("repo-a", "p1")["indexed_at"]
        at2 = state.get_repo_index("repo-a", "p2")["indexed_at"]
        assert at1 is not None and at2 is not None

    def test_indexed_at_not_bled_from_global_repos_for_nondefault(self, state):
        # Simulate a legacy global write (last-writer-wins) then a per-profile index.
        state.upsert_repo("repo-a", "/tmp/repo-a")
        state.update_repo_indexed("repo-a", sha="GLOBAL", branch="main", chunk_count=99)
        state.set_repo_index("repo-a", sha="p2sha", branch="main", chunks=6, profile="p2")
        rec = state.get_repo_index("repo-a", "p2")
        # p2's record must reflect p2, not the global row
        assert rec["sha"] == "p2sha"
        # indexed_at comes from p2's own write, not the global repos timestamp
        assert rec["indexed_at"] == state.get_meta("idx_at:repo-a", "p2")


class TestStalenessCleanup:
    """delete_repo / reset must clear per-profile staleness, or a later index is
    wrongly skipped as 'up to date' against vectors that no longer exist."""

    def test_delete_repo_clears_staleness_all_profiles(self, state):
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        state.set_repo_index("repo-a", sha="s2", branch="main", chunks=6, profile="p2")
        state.set_meta("pending:repo-a", "s1", "p1")
        state.upsert_repo("repo-a", "/tmp/repo-a")

        state.delete_repo("repo-a")

        assert state.get_repo_index_sha("repo-a", "p1") is None
        assert state.get_repo_index_sha("repo-a", "p2") is None
        assert state.get_meta("pending:repo-a", "p1") is None
        assert state.get_repo("repo-a") is None

    def test_delete_repo_does_not_touch_other_repos(self, state):
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        state.set_repo_index("repo-ab", sha="s2", branch="main", chunks=5, profile="p1")
        state.delete_repo("repo-a")
        # Exact-key deletion must not affect a repo whose name is a prefix-superset
        assert state.get_repo_index_sha("repo-ab", "p1") == "s2"

    def test_delete_repo_preserves_global_singletons(self, state):
        state.set_meta("embedding_model", "m1", "p1")
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        state.delete_repo("repo-a")
        assert state.get_meta("embedding_model", "p1") == "m1"
        assert state.get_meta("schema_version") is not None

    def test_clear_all_staleness_clears_everything_except_schema_version(self, state):
        # clear_all_staleness backs `reset --all` (nuke all index data), so it must
        # also drop embedding_model/dims — otherwise doctor reports a phantom index
        # and a later model change dead-ends on "run --full" against empty data.
        state.set_repo_index("repo-a", sha="s1", branch="main", chunks=5, profile="p1")
        state.set_meta("pending:repo-a", "s1", "p2")
        state.set_meta("embedding_model", "m1", "p1")
        state.set_meta("embedding_dims", "1024", "p1")

        state.clear_all_staleness()

        assert state.get_repo_index_sha("repo-a", "p1") is None
        assert state.get_meta("pending:repo-a", "p2") is None
        assert state.get_meta("embedding_model", "p1") is None
        assert state.get_meta("embedding_dims", "p1") is None
        # schema_version is a true global singleton and must survive
        assert state.get_meta("schema_version") is not None


class TestRepoOperations:
    def test_upsert_and_get_repo(self, state):
        state.upsert_repo("my-repo", "/path/to/repo")
        repo = state.get_repo("my-repo")
        assert repo is not None
        assert repo["name"] == "my-repo"
        assert repo["path"] == "/path/to/repo"

    def test_get_repo_missing(self, state):
        assert state.get_repo("nonexistent") is None

    def test_upsert_repo_updates_path(self, state):
        state.upsert_repo("my-repo", "/old/path")
        state.upsert_repo("my-repo", "/new/path")
        repo = state.get_repo("my-repo")
        assert repo["path"] == "/new/path"

    def test_list_repos(self, state):
        state.upsert_repo("alpha", "/a")
        state.upsert_repo("beta", "/b")
        repos = state.list_repos()
        names = [r["name"] for r in repos]
        assert names == ["alpha", "beta"]  # sorted

    def test_list_repos_empty(self, state):
        assert state.list_repos() == []

    def test_delete_repo(self, state):
        state.upsert_repo("my-repo", "/path")
        state.delete_repo("my-repo")
        assert state.get_repo("my-repo") is None

    def test_update_repo_indexed(self, state):
        state.upsert_repo("my-repo", "/path")
        state.update_repo_indexed("my-repo", sha="abc123", branch="main", chunk_count=42)
        repo = state.get_repo("my-repo")
        assert repo["last_indexed_sha"] == "abc123"
        assert repo["last_indexed_branch"] == "main"
        assert repo["chunk_count"] == 42
        assert repo["last_indexed_at"] is not None


class TestSchemaVersion:
    def test_schema_version_set_on_init(self, state):
        assert state.get_meta("schema_version") == "2"

    def test_schema_version_noop_migration_is_silent(self, tmp_path):
        # v1 -> v2 is a structural no-op, so it must auto-bump WITHOUT the scary
        # "full re-index recommended" warning (which would be incorrect here).
        db = StateDB(tmp_path / "test.db")
        db.set_meta("schema_version", "1")
        db.close()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            db2 = StateDB(tmp_path / "test.db")
            assert db2.get_meta("schema_version") == "2"  # bumped
            reindex_warnings = [x for x in w if "re-index" in str(x.message).lower()]
            assert reindex_warnings == []  # no spurious re-index recommendation
            db2.close()


class TestTransaction:
    def test_transaction_commit(self, state):
        with state.transaction():
            state._conn.execute(
                "INSERT INTO repos (name, path) VALUES (?, ?)", ("tx-repo", "/tx")
            )
        assert state.get_repo("tx-repo") is not None

    def test_transaction_rollback(self, state):
        try:
            with state.transaction():
                state._conn.execute(
                    "INSERT INTO repos (name, path) VALUES (?, ?)", ("tx-repo", "/tx")
                )
                raise ValueError("deliberate error")
        except ValueError:
            pass
        assert state.get_repo("tx-repo") is None
