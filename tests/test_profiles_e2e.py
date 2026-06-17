"""End-to-end tests for multiple embedding profiles coexisting on disk.

Proves the user-facing guarantee: switching the active profile in config lets
two embedding models live side by side, queried independently, with per-profile
staleness so indexing one never marks the other fresh.
"""

import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from flowmap.cli import main
from flowmap.config import load_config
from flowmap.state import StateDB
from flowmap.store import VectorStore
from tests.conftest import MockBackend


def _git_init(repo_dir: Path):
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com",
    }
    import os
    env = {**os.environ, **env}
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, capture_output=True, env=env)


def _write_config(config_path: Path, repo_dir: Path, data_dir: Path, active: str):
    config_path.write_text(yaml.dump({
        "repos": [{"name": "test-repo", "path": str(repo_dir)}],
        "data_dir": str(data_dir),
        "embedding": {
            "active": active,
            "profiles": {
                "p1": {"backend": "ollama", "model": "test:mock"},
                "p2": {"backend": "ollama", "model": "test:mock"},
            },
        },
    }))


@pytest.fixture
def two_profile_setup(tmp_path):
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()
    (repo_dir / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef multiply(x, y):\n    return x * y\n"
    )
    _git_init(repo_dir)
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    return config_path, repo_dir, data_dir


def _index(runner, config_path):
    from unittest.mock import patch
    with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
        return runner.invoke(main, ["--config", str(config_path), "index"])


class TestTwoProfilesCoexist:
    def test_index_each_profile_into_its_own_table(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()

        # Index under p1
        _write_config(config_path, repo_dir, data_dir, active="p1")
        r1 = _index(runner, config_path)
        assert r1.exit_code == 0, r1.output
        assert "profile: p1" in r1.output

        # Switch active to p2 and index
        _write_config(config_path, repo_dir, data_dir, active="p2")
        r2 = _index(runner, config_path)
        assert r2.exit_code == 0, r2.output
        assert "profile: p2" in r2.output

        # Both profile tables exist on disk
        cfg = load_config(config_path)
        with VectorStore(cfg.lancedb_path, vector_dims=MockBackend().dims()) as store:
            profiles = store.list_profiles()
        assert "p1" in profiles
        assert "p2" in profiles

    def test_search_returns_results_under_each_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert _index(runner, config_path).exit_code == 0

        for active in ("p1", "p2"):
            _write_config(config_path, repo_dir, data_dir, active=active)
            res = runner.invoke(main, [
                "--config", str(config_path), "search", "add", "--mode", "symbol",
            ])
            assert res.exit_code == 0, res.output
            assert "add" in res.output, f"profile {active}: {res.output}"

    def test_per_profile_staleness_independent(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert _index(runner, config_path).exit_code == 0

        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            sha_p1 = state.get_repo_index_sha("test-repo", "p1")
            sha_p2 = state.get_repo_index_sha("test-repo", "p2")
        assert sha_p1 is not None and sha_p2 is not None
        assert sha_p1 == sha_p2  # same commit, indexed independently

        # Re-indexing p1 with no changes reports up to date (staleness tracked per profile)
        _write_config(config_path, repo_dir, data_dir, active="p1")
        r = _index(runner, config_path)
        assert r.exit_code == 0, r.output
        assert "up to date" in r.output


class TestResetAcrossProfiles:
    def test_reset_repo_purges_all_profiles_and_allows_reindex(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert _index(runner, config_path).exit_code == 0

        # Reset the repo (active profile happens to be p2)
        r = runner.invoke(main, ["--config", str(config_path), "reset", "--repo", "test-repo", "--yes"])
        assert r.exit_code == 0, r.output

        # Both profile tables must have zero chunks for the repo
        cfg = load_config(config_path)
        with VectorStore(cfg.lancedb_path, vector_dims=MockBackend().dims()) as store:
            for prof in ("p1", "p2"):
                stats = store.get_stats(profile=prof, known_repos=["test-repo"])
                assert stats["repos"].get("test-repo", 0) == 0, f"{prof} still has chunks"

        # Staleness cleared → a fresh index actually re-indexes (NOT 'up to date')
        _write_config(config_path, repo_dir, data_dir, active="p1")
        r2 = _index(runner, config_path)
        assert r2.exit_code == 0, r2.output
        assert "up to date" not in r2.output


class TestCrossDimProfiles:
    """The headline use case: two profiles whose embedding models have DIFFERENT
    vector dims (e.g. qwen 0.6b=1024 vs 4b=2560) coexisting on disk."""

    def test_cross_dim_profiles_end_to_end(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()

        def index_with(active, dims):
            _write_config(config_path, repo_dir, data_dir, active=active)
            with patch("flowmap.embeddings.create_backend", return_value=MockBackend(dims=dims)):
                return runner.invoke(main, ["--config", str(config_path), "index"])

        # p1 at 1024 dims (qwen 0.6b), p2 at 2560 dims (qwen 4b) — the real
        # headline case, coexisting in separate tables.
        assert index_with("p1", 1024).exit_code == 0
        assert index_with("p2", 2560).exit_code == 0

        # Introspection commands that iterate profiles must not blow up on dim diff.
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert runner.invoke(main, ["--config", str(config_path), "status"]).exit_code == 0
        assert runner.invoke(main, ["--config", str(config_path), "repos", "list"]).exit_code == 0

        # Search under each profile with its matching dims returns that profile's data.
        for active, dims in (("p1", 1024), ("p2", 2560)):
            _write_config(config_path, repo_dir, data_dir, active=active)
            with patch("flowmap.embeddings.create_backend", return_value=MockBackend(dims=dims)):
                res = runner.invoke(main, ["--config", str(config_path), "search", "add", "--mode", "semantic"])
            assert res.exit_code == 0, f"{active}: {res.output}"

        # reset --all across mixed-dim tables must succeed.
        assert runner.invoke(main, ["--config", str(config_path), "reset", "--all", "--yes"]).exit_code == 0


class TestIncrementalAndRecoveryUnderProfile:
    def _commit(self, repo_dir):
        import os
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"}
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change"], cwd=repo_dir, capture_output=True, env=env)

    def test_incremental_reindex_under_nondefault_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        # Modify a file + commit, then re-index → exercises compute_incremental body.
        (repo_dir / "math_utils.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef subtract(x, y):\n    return x - y\n"
        )
        self._commit(repo_dir)
        r = _index(runner, config_path)
        assert r.exit_code == 0, r.output
        assert "up to date" not in r.output  # actually re-indexed incrementally

        # New symbol searchable under p1; p2 untouched (never indexed).
        s = runner.invoke(main, ["--config", str(config_path), "search", "subtract", "--mode", "symbol"])
        assert "subtract" in s.output

    def test_pending_marker_forces_full_reindex(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        # Simulate an interrupted prior index by setting the pending marker.
        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            state.set_meta("pending:test-repo", "deadbeef", "p1")

        r = _index(runner, config_path)
        assert r.exit_code == 0, r.output
        assert "interrupted previous index" in r.output


class TestGlobalMirrorNotPoisonedByNonDefault:
    def test_nondefault_index_does_not_write_global_repos_row(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Index only p1 (non-default) — global repos row must stay unindexed.
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            info = state.get_repo("test-repo")
            # Row exists (path upserted) but no global last_indexed_sha mirror.
            assert info is not None
            assert not info.get("last_indexed_sha")
            # Per-profile record is populated.
            assert state.get_repo_index("test-repo", "p1")["sha"]


class TestResetBenchmarks:
    def test_spares_config_profiles_and_clears_dropped_staleness(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Index config profile p1, plus a stray on-disk "bench" profile not in config.
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        # Create a stray profile by writing a config that has it, indexing, then
        # reverting config to p1/p2 only.
        config_path.write_text(__import__("yaml").dump({
            "repos": [{"name": "test-repo", "path": str(repo_dir)}],
            "data_dir": str(data_dir),
            "embedding": {"active": "bench", "profiles": {"bench": {"backend": "ollama", "model": "test:mock"}}},
        }))
        assert _index(runner, config_path).exit_code == 0
        _write_config(config_path, repo_dir, data_dir, active="p1")  # bench now NOT in config

        r = runner.invoke(main, ["--config", str(config_path), "reset", "--benchmarks", "--yes"])
        assert r.exit_code == 0, r.output

        cfg = load_config(config_path)
        with VectorStore(cfg.lancedb_path, vector_dims=MockBackend().dims()) as store:
            profiles = store.list_profiles()
        assert "p1" in profiles          # config profile spared
        assert "bench" not in profiles   # stray benchmark dropped

        # Dropped profile's staleness cleared (so re-add re-indexes, not "up to date")
        with StateDB(cfg.db_path) as state:
            assert state.get_repo_index_sha("test-repo", "bench") is None


class TestReadCommandsUseActiveProfile:
    """map / symbols / doctor / cat / history must read the active profile, not
    always the default table."""

    def test_map_and_symbols_use_active_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Index ONLY p2 — default table never created.
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert _index(runner, config_path).exit_code == 0

        # Under active p2: map + symbols return data.
        m = runner.invoke(main, ["--config", str(config_path), "map"])
        assert m.exit_code == 0
        assert "test-repo" in m.output
        s = runner.invoke(main, ["--config", str(config_path), "symbols", "add"])
        assert s.exit_code == 0
        assert "add" in s.output

        # Under active p1 (never indexed): clean empty, no crash.
        _write_config(config_path, repo_dir, data_dir, active="p1")
        m1 = runner.invoke(main, ["--config", str(config_path), "map"])
        assert m1.exit_code == 0
        assert "No indexed data" in m1.output

    def test_doctor_reports_active_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p2")
        assert _index(runner, config_path).exit_code == 0

        from unittest.mock import patch
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
            d = runner.invoke(main, ["--config", str(config_path), "doctor"])
        assert "profile: p2" in d.output
        assert "EMPTY" not in d.output  # p2 IS indexed


class TestReposListProfileAware:
    def test_repos_list_reflects_active_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Index only under p1.
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        # Active p1 → indexed; active p2 → not indexed (not last-writer-wins).
        _write_config(config_path, repo_dir, data_dir, active="p1")
        r1 = runner.invoke(main, ["--config", str(config_path), "repos", "list"])
        assert "profile: p1" in r1.output
        assert "indexed" in r1.output and "not indexed" not in r1.output

        _write_config(config_path, repo_dir, data_dir, active="p2")
        r2 = runner.invoke(main, ["--config", str(config_path), "repos", "list"])
        assert "not indexed" in r2.output


class TestModelDriftWarning:
    """If a profile's config is pointed at a different (same-dim) model than it was
    indexed with, search silently queries the wrong vectors — warn the user."""

    def test_search_warns_when_model_drifts(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")

        # Index p1 with "modelA"
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="modelA")):
            assert runner.invoke(main, ["--config", str(config_path), "index"]).exit_code == 0

        # Search p1 with "modelB" (same dims) → drift warning
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="modelB")):
            res = runner.invoke(main, ["--config", str(config_path), "search", "add", "--mode", "semantic"])
        assert res.exit_code == 0
        assert "modelA" in res.output and "modelB" in res.output

        # Suppressed under --format json (machine output stays clean)
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="modelB")):
            res2 = runner.invoke(main, ["--config", str(config_path), "search", "add", "--mode", "semantic", "--format", "json"])
        assert "modelA" not in res2.output


class TestPreProfileDbUpgrade:
    """A DB written by the pre-profile version has chunks in the default `code_index`
    table and a populated `repos` row, but NO per-profile `idx_*` meta. The new code
    must still report it indexed (via the default fallback) and search it."""

    def test_legacy_db_reads_via_default_fallback(self, two_profile_setup):
        import yaml as _yaml
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Flat (legacy) config → default profile.
        config_path.write_text(_yaml.dump({
            "repos": [{"name": "test-repo", "path": str(repo_dir)}],
            "data_dir": str(data_dir),
            "embedding": {"backend": "ollama", "model": "test:mock"},
        }))
        assert _index(runner, config_path).exit_code == 0

        # Simulate a pre-profile DB: drop the per-profile idx_* meta, leaving only
        # the global repos row + the code_index table (exactly the old schema state).
        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            state._conn.execute("DELETE FROM meta WHERE key GLOB 'idx_*'")
            state._conn.commit()
            assert state.get_repo_index_sha("test-repo", "default") is None  # no per-profile meta
            assert state.get_repo_index("test-repo", "default")["sha"]        # fallback to repos row

        # status reports indexed via fallback; search still works.
        st = runner.invoke(main, ["--config", str(config_path), "status"])
        assert st.exit_code == 0
        assert "not indexed" not in st.output
        s = runner.invoke(main, ["--config", str(config_path), "search", "add", "--mode", "symbol"])
        assert s.exit_code == 0 and "add" in s.output


class TestModelChangeGuardSkippedWhenEmpty:
    def test_reset_repo_then_switch_model_no_deadend(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")

        # Index p1 with modelA.
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="modelA")):
            assert runner.invoke(main, ["--config", str(config_path), "index"]).exit_code == 0

        # reset --repo empties p1's table but leaves embedding_model=modelA meta.
        assert runner.invoke(main, ["--config", str(config_path), "reset", "--repo", "test-repo", "--yes"]).exit_code == 0

        # Switching the model and re-indexing must NOT dead-end on "model changed".
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="modelB")):
            r = runner.invoke(main, ["--config", str(config_path), "index"])
        assert r.exit_code == 0, r.output
        assert "changed" not in r.output.lower()


class TestStaleCleanupFailureKeepsPending:
    def test_pending_marker_retained_when_stale_cleanup_fails(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")

        # Force stale cleanup to fail during the full index.
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()), \
             patch("flowmap.store.VectorStore.delete_stale_files", return_value=False):
            r = runner.invoke(main, ["--config", str(config_path), "index"])
        assert r.exit_code == 0, r.output

        # Upsert landed, but the pending marker must remain so the next run forces full.
        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            assert state.get_meta("pending:test-repo", "p1")  # non-empty


class TestIncrementalStaleCleanupFailureKeepsPending:
    def test_pending_retained_when_incremental_cleanup_fails(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        # Modify a file (removes a symbol) + commit → triggers the modified-file /
        # delete_stale_chunks path on the next (incremental) index.
        (repo_dir / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
        import os
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"}
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "shrink"], cwd=repo_dir, capture_output=True, env=env)

        # Force incremental stale-cleanup to fail.
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()), \
             patch("flowmap.store.VectorStore.delete_stale_chunks", return_value=False):
            r = runner.invoke(main, ["--config", str(config_path), "index"])
        assert r.exit_code == 0, r.output

        # Pending marker must remain so the next run forces a clean full reindex.
        cfg = load_config(config_path)
        with StateDB(cfg.db_path) as state:
            assert state.get_meta("pending:test-repo", "p1")  # non-empty


class TestResetAllClearsModelMeta:
    def test_doctor_empty_and_reindex_clean_after_reset_all(self, two_profile_setup):
        from unittest.mock import patch
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        assert runner.invoke(main, ["--config", str(config_path), "reset", "--all", "--yes"]).exit_code == 0

        # doctor must report EMPTY, not a phantom indexed model with 0 chunks.
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
            d = runner.invoke(main, ["--config", str(config_path), "doctor"])
        assert "EMPTY" in d.output or "No index data" in d.output
        assert "Model:" not in d.output

        # Re-indexing (even with a different model) must NOT dead-end on "run --full".
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend(model_name="differentModel")):
            r = runner.invoke(main, ["--config", str(config_path), "index"])
        assert r.exit_code == 0, r.output
        assert "model" not in r.output.lower() or "changed" not in r.output.lower()


class TestUnindexedProfileWarning:
    def test_search_warns_when_active_profile_not_indexed(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        # Index only p1, then search under p2 (never indexed)
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0

        _write_config(config_path, repo_dir, data_dir, active="p2")
        res = runner.invoke(main, [
            "--config", str(config_path), "search", "add", "--mode", "symbol",
        ])
        assert res.exit_code == 0
        assert "not indexed" in res.output

    def test_status_shows_active_profile(self, two_profile_setup):
        config_path, repo_dir, data_dir = two_profile_setup
        runner = CliRunner()
        _write_config(config_path, repo_dir, data_dir, active="p1")
        assert _index(runner, config_path).exit_code == 0
        res = runner.invoke(main, ["--config", str(config_path), "status"])
        assert res.exit_code == 0, res.output
        assert "profile: p1" in res.output
