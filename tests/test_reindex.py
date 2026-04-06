"""Tests for incremental reindexing — git diff parsing, branch detection, change handling."""

from flowmap.reindex import (
    FileChange,
    GitStatus,
    IncrementalResult,
    get_changed_files,
    should_full_reindex,
)


# ---------------------------------------------------------------------------
# should_full_reindex
# ---------------------------------------------------------------------------

def test_first_index_requires_full():
    current = GitStatus(sha="abc123", branch="main")
    needs_full, reason = should_full_reindex(None, None, current)
    assert needs_full is True
    assert "first index" in reason


def test_same_sha_is_up_to_date():
    current = GitStatus(sha="abc123", branch="main")
    needs_full, reason = should_full_reindex("abc123", "main", current)
    assert needs_full is False
    assert "up to date" in reason


def test_branch_change_requires_full():
    current = GitStatus(sha="def456", branch="dev")
    needs_full, reason = should_full_reindex("abc123", "main", current)
    assert needs_full is True
    assert "branch changed" in reason
    assert "main" in reason
    assert "dev" in reason


def test_same_branch_different_sha_is_incremental():
    current = GitStatus(sha="def456", branch="main")
    needs_full, reason = should_full_reindex("abc123", "main", current)
    assert needs_full is False


def test_no_stored_branch_with_different_sha():
    """First time branch tracking — stored branch is empty."""
    current = GitStatus(sha="def456", branch="main")
    needs_full, reason = should_full_reindex("abc123", "", current)
    assert needs_full is False  # empty stored branch = don't know, allow incremental


def test_none_stored_branch_with_different_sha():
    current = GitStatus(sha="def456", branch="main")
    needs_full, reason = should_full_reindex("abc123", None, current)
    assert needs_full is False  # None stored branch = don't know, allow incremental


# ---------------------------------------------------------------------------
# FileChange parsing
# ---------------------------------------------------------------------------

def test_file_change_added():
    c = FileChange(status="A", path="src/new_file.py")
    assert c.status == "A"
    assert c.path == "src/new_file.py"
    assert c.old_path is None


def test_file_change_modified():
    c = FileChange(status="M", path="src/existing.py")
    assert c.status == "M"


def test_file_change_deleted():
    c = FileChange(status="D", path="src/removed.py")
    assert c.status == "D"


def test_file_change_renamed():
    c = FileChange(status="R", path="src/new_name.py", old_path="src/old_name.py")
    assert c.status == "R"
    assert c.old_path == "src/old_name.py"
    assert c.path == "src/new_name.py"


# ---------------------------------------------------------------------------
# IncrementalResult
# ---------------------------------------------------------------------------

def test_incremental_result_skipped():
    r = IncrementalResult(mode="skipped", reason="no changes")
    assert r.mode == "skipped"
    assert r.added == 0
    assert r.total_chunks == 0


def test_incremental_result_with_changes():
    r = IncrementalResult(
        mode="incremental", reason="abc→def",
        added=3, modified=2, deleted=1, renamed=1, total_chunks=45,
    )
    assert r.added == 3
    assert r.modified == 2
    assert r.deleted == 1
    assert r.renamed == 1
    assert r.total_chunks == 45


# ---------------------------------------------------------------------------
# GitStatus
# ---------------------------------------------------------------------------

def test_git_status_dataclass():
    gs = GitStatus(sha="abc123def", branch="feature/auth")
    assert gs.sha == "abc123def"
    assert gs.branch == "feature/auth"


# ---------------------------------------------------------------------------
# get_changed_files (requires git, tests against this repo)
# ---------------------------------------------------------------------------

def test_get_changed_files_invalid_sha(tmp_path):
    """Invalid old SHA should return None (triggers full reindex)."""
    import subprocess
    # Create a temp git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

    result = get_changed_files(str(tmp_path), "0000000000000000000000000000000000000000", "HEAD")
    assert result is None  # invalid SHA → full reindex


def test_get_changed_files_with_changes(tmp_path):
    """Detect added/modified files between commits."""
    import subprocess
    # Create a temp git repo with two commits
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

    (tmp_path / "file1.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, capture_output=True)
    sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()

    (tmp_path / "file2.txt").write_text("world")
    (tmp_path / "file1.txt").write_text("hello modified")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=tmp_path, capture_output=True)

    changes = get_changed_files(str(tmp_path), sha1, "HEAD")
    assert changes is not None
    statuses = {c.status for c in changes}
    paths = {c.path for c in changes}
    assert "A" in statuses  # file2.txt added
    assert "M" in statuses  # file1.txt modified
    assert "file2.txt" in paths
    assert "file1.txt" in paths


def test_get_changed_files_no_changes(tmp_path):
    """Same SHA should return empty list."""
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()

    changes = get_changed_files(str(tmp_path), sha, "HEAD")
    assert changes is not None
    assert len(changes) == 0


# ---------------------------------------------------------------------------
# compute_incremental: mutation ordering (upsert before delete)
# ---------------------------------------------------------------------------

def test_incremental_upsert_before_delete_for_modified_files():
    """Verify upsert runs BEFORE delete for modified files (no data-loss window)."""
    from unittest.mock import MagicMock, call
    from flowmap.reindex import compute_incremental

    store = MagicMock()
    store.get_stats.return_value = {"total": 5, "repos": {"repo1": 5}}
    state = MagicMock()
    state.get_meta.return_value = None

    backend = MagicMock()
    backend.embed_documents.return_value = [[0.1] * 32]

    # Fake indexer returns one chunk for the modified file
    def fake_indexer(repo_path, repo_name, file_list):
        return [{"id": "new_chunk_1", "file": "modified.py", "text": "def foo(): pass",
                 "repo": repo_name, "file_name": "modified.py", "extension": ".py"}]

    # Mock git diff to return one modified file
    import flowmap.reindex as reindex_mod
    original_get_changed = reindex_mod.get_changed_files

    def mock_get_changed(repo_path, old_sha, new_sha):
        from flowmap.reindex import FileChange
        return [FileChange(status="M", path="modified.py")]

    reindex_mod.get_changed_files = mock_get_changed
    try:
        current = GitStatus(sha="new_sha", branch="main")
        result = compute_incremental(
            repo_path="/tmp/fake",
            repo_name="repo1",
            stored_sha="old_sha",
            current=current,
            indexer_fn=fake_indexer,
            store=store,
            embedding_backend=backend,
            state_db=state,
        )
    finally:
        reindex_mod.get_changed_files = original_get_changed

    assert result.mode == "incremental"
    # CRITICAL: upsert_chunks must be called BEFORE delete_stale_chunks
    upsert_call_idx = None
    delete_stale_call_idx = None
    for i, c in enumerate(store.method_calls):
        if c[0] == "upsert_chunks":
            upsert_call_idx = i
        if c[0] == "delete_stale_chunks":
            delete_stale_call_idx = i

    assert upsert_call_idx is not None, "upsert_chunks was not called"
    assert delete_stale_call_idx is not None, "delete_stale_chunks was not called"
    assert upsert_call_idx < delete_stale_call_idx, \
        f"upsert_chunks (call #{upsert_call_idx}) must run BEFORE delete_stale_chunks (call #{delete_stale_call_idx})"


def test_incremental_embed_failure_preserves_data():
    """If embedding fails during incremental, no store mutations happen."""
    from unittest.mock import MagicMock
    from flowmap.reindex import compute_incremental

    store = MagicMock()
    state = MagicMock()
    state.get_meta.return_value = None

    backend = MagicMock()
    backend.embed_documents.side_effect = ConnectionError("Ollama crashed")

    def fake_indexer(repo_path, repo_name, file_list):
        return [{"id": "c1", "file": "a.py", "text": "x", "repo": repo_name,
                 "file_name": "a.py", "extension": ".py"}]

    import flowmap.reindex as reindex_mod
    original_get_changed = reindex_mod.get_changed_files

    def mock_get_changed(repo_path, old_sha, new_sha):
        from flowmap.reindex import FileChange
        return [FileChange(status="M", path="a.py")]

    reindex_mod.get_changed_files = mock_get_changed
    try:
        current = GitStatus(sha="new_sha", branch="main")
        try:
            compute_incremental(
                repo_path="/tmp/fake", repo_name="repo1",
                stored_sha="old_sha", current=current,
                indexer_fn=fake_indexer, store=store,
                embedding_backend=backend, state_db=state,
            )
        except ConnectionError:
            pass
    finally:
        reindex_mod.get_changed_files = original_get_changed

    # CRITICAL: no store mutations when embedding fails
    store.upsert_chunks.assert_not_called()
    store.delete_by_file.assert_not_called()
    store.delete_stale_chunks.assert_not_called()
