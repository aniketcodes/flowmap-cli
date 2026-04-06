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

    def test_schema_version_mismatch_warns(self, tmp_path):
        # Create DB with version 1
        db = StateDB(tmp_path / "test.db")
        db.set_meta("schema_version", "1")
        db.close()

        # Re-open — should warn and auto-migrate
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            db2 = StateDB(tmp_path / "test.db")
            assert len(w) == 1
            assert "migrated" in str(w[0].message).lower()
            # Version should be auto-migrated
            assert db2.get_meta("schema_version") == "2"
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
