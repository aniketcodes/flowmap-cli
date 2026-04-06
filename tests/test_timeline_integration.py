"""Integration tests for build_timeline and run_index orchestrators."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from flowmap.history.timeline import Timeline, build_timeline
from flowmap.services.indexing import IndexResult, run_index
from flowmap.store import SearchResult
from tests.conftest import FakeCommit, FakeHistory


def _make_search_result(repo: str, file: str, symbol: str, line: int = 1) -> SearchResult:
    return SearchResult(
        repo=repo, file=file, start_line=line, end_line=line + 10,
        text=f"def {symbol}(): pass", symbol_name=symbol, chunk_type="function",
        signature=f"def {symbol}():", parent_symbol="", parent_signature="",
        language="python", score=1.0,
    )


class TestBuildTimeline:
    def test_empty_index(self):
        store = MagicMock()
        store.search_symbol.return_value = []
        tl = build_timeline("query", {}, store)
        assert tl.entries == []
        assert tl.scoped_files == []

    def test_scopes_from_symbol_search(self):
        store = MagicMock()
        store.search_symbol.return_value = [
            _make_search_result("repo1", "a.py", "foo"),
        ]

        with patch("flowmap.history.timeline.get_file_history") as mock_hist, \
             patch("flowmap.history.timeline.pickaxe_search") as mock_pick:
            mock_hist.return_value = None
            mock_pick.return_value = []
            tl = build_timeline("foo", {"repo1": "/tmp/repo1"}, store)

        assert "repo1/a.py" in tl.scoped_files

    def test_repo_filter(self):
        store = MagicMock()
        store.search_symbol.return_value = []
        tl = build_timeline("query", {"repo1": "/r1", "repo2": "/r2"}, store, repo_filter="repo1")
        store.search_symbol.assert_called_once_with("query", repo_filter="repo1", limit=10)

    def test_respects_limit(self):
        store = MagicMock()
        store.search_symbol.return_value = [
            _make_search_result("repo1", "a.py", "foo"),
        ]


        commits = [FakeCommit(f"sha{i}", "dev", f"2026-01-{i+1:02d}T00:00:00+00:00", f"msg{i}") for i in range(10)]
        fake_hist = FakeHistory(file="a.py", commits=commits)

        with patch("flowmap.history.timeline.get_file_history") as mock_hist, \
             patch("flowmap.history.timeline.pickaxe_search") as mock_pick, \
             patch("flowmap.history.timeline._diff_commit") as mock_diff:
            mock_hist.return_value = fake_hist
            mock_pick.return_value = []
            mock_diff.return_value = []
            tl = build_timeline("foo", {"repo1": "/tmp/repo1"}, store, limit=3)

        assert len(tl.entries) <= 3

    def test_diff_failure_doesnt_crash(self):
        """If structural diff raises, the entry should still appear (without changes)."""
        store = MagicMock()
        store.search_symbol.return_value = [
            _make_search_result("repo1", "a.py", "foo"),
        ]


        fake_hist = FakeHistory(file="a.py", commits=[
            FakeCommit("abc123", "dev", "2026-01-15T00:00:00+00:00", "test commit"),
        ])

        with patch("flowmap.history.timeline.get_file_history") as mock_hist, \
             patch("flowmap.history.timeline.pickaxe_search") as mock_pick, \
             patch("flowmap.history.timeline._diff_commit") as mock_diff:
            mock_hist.return_value = fake_hist
            mock_pick.return_value = []
            mock_diff.side_effect = RuntimeError("git broke")
            tl = build_timeline("foo", {"repo1": "/tmp/repo1"}, store, limit=5)

        # Should have at least one entry, even though diff failed
        assert len(tl.entries) >= 1
        # Entry should have empty changes (diff failed)
        assert tl.entries[0].changes == []


    def test_timeout_doesnt_crash(self):
        """If structural diffs time out, timeline still returns entries without crashing."""
        from concurrent.futures import Future

        store = MagicMock()
        store.search_symbol.return_value = [
            _make_search_result("repo1", "a.py", "foo"),
        ]


        fake_hist = FakeHistory(file="a.py", commits=[
            FakeCommit("abc123", "dev", "2026-01-15T00:00:00+00:00", "slow commit"),
        ])

        # Create a fake future that is NOT done (simulating a timed-out task)
        fake_future = Future()  # not resolved — future.done() == False

        def mock_as_completed(futures, timeout=None):
            """Raise TimeoutError immediately — simulates all futures timing out."""
            raise TimeoutError("timed out")

        def mock_submit(fn, *args, **kwargs):
            """Return our fake (unresolved) future instead of running the function."""
            return fake_future

        with patch("flowmap.history.timeline.get_file_history") as mock_hist, \
             patch("flowmap.history.timeline.pickaxe_search") as mock_pick, \
             patch("flowmap.history.timeline._diff_commit", return_value=[]), \
             patch("flowmap.history.timeline.as_completed", side_effect=mock_as_completed), \
             patch("flowmap.history.timeline.ThreadPoolExecutor") as mock_executor_cls:
            mock_hist.return_value = fake_hist
            mock_pick.return_value = []

            # Make the executor context manager return a mock with our submit
            mock_executor = MagicMock()
            mock_executor.submit.side_effect = mock_submit
            mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

            tl = build_timeline("foo", {"repo1": "/tmp/repo1"}, store, limit=5)

        # Should not crash — timed-out entries added as metadata-only
        assert isinstance(tl.entries, list)
        assert len(tl.entries) >= 1
        # The entry should have no structural changes (timed out)
        assert tl.entries[0].changes == []


class TestRunIndex:
    def _make_mocks(self, tmp_path):
        """Create mock store, state, and backend for run_index tests."""
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        (repo_dir / "hello.py").write_text("def hello():\n    pass\n")

        @dataclass
        class FakeRepo:
            name: str
            path: str
            def resolved_path(self):
                from pathlib import Path
                return Path(self.path)

        target = FakeRepo(name="my-repo", path=str(repo_dir))
        store = MagicMock()
        store.get_stats.return_value = {"total": 5, "repos": {"my-repo": 5}}
        state = MagicMock()
        state.get_repo.return_value = None  # not yet indexed
        state.get_meta.return_value = None

        backend = MagicMock()
        backend.model_name.return_value = "test-model"
        backend.dims.return_value = 32
        backend.embed_documents.return_value = [[0.1] * 32]  # one embedding

        return target, store, state, backend

    def test_full_index(self, tmp_path):
        target, store, state, backend = self._make_mocks(tmp_path)

        with patch("flowmap.services.indexing.get_git_status") as mock_git, \
             patch("flowmap.services.indexing.index_repo") as mock_idx:
            mock_git.return_value = MagicMock(sha="abc123", branch="main")
            mock_idx.return_value = [{"text": "def hello(): pass", "id": "c1", "repo": "my-repo",
                                       "file": "hello.py", "file_name": "hello.py", "extension": ".py"}]
            backend.embed_documents.return_value = [[0.1] * 32]

            results = run_index(store, state, backend, [target], full=True)

        assert len(results) == 1
        assert results[0].mode == "full"
        # Upsert-first pattern: upsert is called, then stale files cleaned up
        store.upsert_chunks.assert_called_once()
        store.delete_stale_files.assert_called_once()

    def test_skip_up_to_date(self, tmp_path):
        target, store, state, backend = self._make_mocks(tmp_path)
        state.get_repo.return_value = {"last_indexed_sha": "abc123", "last_indexed_branch": "main"}

        with patch("flowmap.services.indexing.get_git_status") as mock_git, \
             patch("flowmap.services.indexing.should_full_reindex") as mock_should:
            mock_git.return_value = MagicMock(sha="abc123", branch="main")
            mock_should.return_value = (False, "already up to date")

            results = run_index(store, state, backend, [target])

        assert results[0].mode == "skipped"

    def test_missing_path(self, tmp_path):
        @dataclass
        class FakeRepo:
            name: str
            path: str
            def resolved_path(self):
                from pathlib import Path
                return Path(self.path)

        target = FakeRepo(name="gone-repo", path=str(tmp_path / "nonexistent"))
        store = MagicMock()
        state = MagicMock()
        backend = MagicMock()

        results = run_index(store, state, backend, [target])
        assert results[0].mode == "error"
        assert "path not found" in results[0].message
