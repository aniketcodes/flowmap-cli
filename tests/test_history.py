"""Tests for temporal code intelligence — git_ops, structural_diff, timeline."""

import subprocess

from flowmap.history.git_ops import (
    CommitInfo,
    FileHistory,
    get_file_at_commit,
    get_file_history,
    pickaxe_search,
)
from flowmap.history.structural_diff import SymbolChange, structural_diff


# ---------------------------------------------------------------------------
# Helper: create a git repo with commits
# ---------------------------------------------------------------------------

def _init_repo(tmp_path):
    """Create an initialized git repo in tmp_path."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True)


def _commit(tmp_path, message="commit"):
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=tmp_path, capture_output=True)


def _get_sha(tmp_path):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# git_ops: get_file_history
# ---------------------------------------------------------------------------

def test_get_file_history_basic(tmp_path):
    """Get commit history for a file with multiple commits."""
    _init_repo(tmp_path)

    (tmp_path / "app.py").write_text("v1")
    _commit(tmp_path, "first commit")

    (tmp_path / "app.py").write_text("v2")
    _commit(tmp_path, "second commit")

    (tmp_path / "app.py").write_text("v3")
    _commit(tmp_path, "third commit")

    history = get_file_history(str(tmp_path), "app.py")
    assert history is not None
    assert history.file == "app.py"
    assert len(history.commits) == 3
    assert history.commits[0].message == "third commit"  # newest first
    assert history.commits[2].message == "first commit"


def test_get_file_history_limit(tmp_path):
    _init_repo(tmp_path)

    for i in range(5):
        (tmp_path / "app.py").write_text(f"v{i}")
        _commit(tmp_path, f"commit {i}")

    history = get_file_history(str(tmp_path), "app.py", limit=3)
    assert history is not None
    assert len(history.commits) == 3


def test_get_file_history_follows_renames(tmp_path):
    """--follow should track file renames."""
    _init_repo(tmp_path)

    (tmp_path / "old.py").write_text("content")
    _commit(tmp_path, "create old.py")

    subprocess.run(["git", "mv", "old.py", "new.py"], cwd=tmp_path, capture_output=True)
    _commit(tmp_path, "rename to new.py")

    history = get_file_history(str(tmp_path), "new.py")
    assert history is not None
    assert len(history.commits) >= 2  # should include pre-rename commit


def test_get_file_history_nonexistent_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "app.py").write_text("x")
    _commit(tmp_path, "init")

    history = get_file_history(str(tmp_path), "nonexistent.py")
    assert history is not None
    assert len(history.commits) == 0


def test_get_file_history_not_a_git_repo(tmp_path):
    history = get_file_history(str(tmp_path), "app.py")
    assert history is None


# ---------------------------------------------------------------------------
# git_ops: get_file_at_commit
# ---------------------------------------------------------------------------

def test_get_file_at_commit_basic(tmp_path):
    _init_repo(tmp_path)

    (tmp_path / "app.py").write_text("version_one")
    _commit(tmp_path, "v1")
    sha1 = _get_sha(tmp_path)

    (tmp_path / "app.py").write_text("version_two")
    _commit(tmp_path, "v2")

    content = get_file_at_commit(str(tmp_path), "app.py", sha1)
    assert content == "version_one"


def test_get_file_at_commit_not_exists(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "other.py").write_text("x")
    _commit(tmp_path, "init")
    sha = _get_sha(tmp_path)

    content = get_file_at_commit(str(tmp_path), "nonexistent.py", sha)
    assert content is None


def test_get_file_at_commit_invalid_sha(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "app.py").write_text("x")
    _commit(tmp_path, "init")

    content = get_file_at_commit(str(tmp_path), "app.py", "0" * 40)
    assert content is None


# ---------------------------------------------------------------------------
# git_ops: pickaxe_search
# ---------------------------------------------------------------------------

def test_pickaxe_search_finds_string(tmp_path):
    _init_repo(tmp_path)

    (tmp_path / "app.py").write_text("hello world")
    _commit(tmp_path, "add hello")

    (tmp_path / "app.py").write_text("goodbye world")
    _commit(tmp_path, "change to goodbye")

    results = pickaxe_search(str(tmp_path), "hello")
    assert len(results) >= 1
    messages = [c.message for c in results]
    assert any("hello" in m or "goodbye" in m for m in messages)


def test_pickaxe_search_no_results(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "app.py").write_text("nothing special")
    _commit(tmp_path, "init")

    results = pickaxe_search(str(tmp_path), "xyznonexistent123")
    assert results == []


def test_pickaxe_search_not_a_repo(tmp_path):
    results = pickaxe_search(str(tmp_path), "hello")
    assert results == []


def test_pickaxe_search_dash_query(tmp_path):
    """Query starting with dash doesn't get interpreted as git flag."""
    _init_repo(tmp_path)
    (tmp_path / "app.py").write_text("x = '--all is a flag'")
    _commit(tmp_path, "add app")
    # Should not crash — git should treat --all as the -S search string, not a flag
    results = pickaxe_search(str(tmp_path), "--all")
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# structural_diff: symbol-level comparison
# ---------------------------------------------------------------------------

def test_structural_diff_added_function():
    old = ""
    new = "def hello():\n    return 'hi'\n"
    changes = structural_diff(None, new, ".py")
    assert len(changes) >= 1
    added = [c for c in changes if c.change_type == "added"]
    assert len(added) >= 1
    assert added[0].symbol_name == "hello"
    assert added[0].significance == "major"


def test_structural_diff_removed_function():
    old = "def hello():\n    return 'hi'\n"
    changes = structural_diff(old, None, ".py")
    assert len(changes) >= 1
    removed = [c for c in changes if c.change_type == "removed"]
    assert len(removed) >= 1
    assert removed[0].symbol_name == "hello"


def test_structural_diff_signature_changed():
    old = "def process(x):\n    return x\n"
    new = "def process(x, y=None):\n    return x\n"
    changes = structural_diff(old, new, ".py")
    sig_changes = [c for c in changes if c.change_type == "signature_changed"]
    assert len(sig_changes) == 1
    assert sig_changes[0].symbol_name == "process"
    assert sig_changes[0].significance == "major"


def test_structural_diff_body_changed():
    old = "def process(x):\n    return x\n"
    new = "def process(x):\n    return x + 1\n"
    changes = structural_diff(old, new, ".py")
    body_changes = [c for c in changes if c.change_type == "body_changed"]
    assert len(body_changes) == 1
    assert body_changes[0].symbol_name == "process"


def test_structural_diff_unchanged():
    code = "def process(x):\n    return x\n"
    changes = structural_diff(code, code, ".py")
    assert len(changes) == 0  # unchanged symbols are skipped


def test_structural_diff_both_none():
    changes = structural_diff(None, None, ".py")
    assert changes == []


def test_structural_diff_unsupported_extension():
    """Unsupported extensions produce no parseable symbols → empty diff."""
    old = "# some markdown\n## heading\n"
    new = "# different markdown\n## heading\n"
    changes = structural_diff(old, new, ".md")
    assert changes == []


def test_structural_diff_multiple_functions():
    old = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    new = "def foo():\n    return 1\n\ndef baz():\n    pass\n"
    changes = structural_diff(old, new, ".py")
    types = {c.symbol_name: c.change_type for c in changes}
    assert types.get("bar") == "removed"
    assert types.get("baz") == "added"
    assert types.get("foo") == "body_changed"


# ---------------------------------------------------------------------------
# CommitInfo / FileHistory dataclass basics
# ---------------------------------------------------------------------------

def test_commit_info_fields():
    c = CommitInfo(sha="abc123", author="John", date="2026-01-01T00:00:00+00:00", message="fix bug")
    assert c.sha == "abc123"
    assert c.author == "John"


def test_file_history_empty():
    h = FileHistory(file="app.py")
    assert h.commits == []


def test_symbol_change_fields():
    sc = SymbolChange(
        symbol_name="validate",
        change_type="signature_changed",
        old_signature="def validate(x)",
        new_signature="def validate(x, strict=False)",
        significance="major",
    )
    assert sc.change_type == "signature_changed"
    assert sc.significance == "major"


# ---------------------------------------------------------------------------
# git_ops: _parse_log_output delimiter handling
# ---------------------------------------------------------------------------

def test_parse_log_output_pipe_in_author(tmp_path):
    """Author name containing | does not corrupt parsed fields."""
    _init_repo(tmp_path)
    # Set author with pipe in name
    subprocess.run(["git", "config", "user.name", "Joe | Dev"], cwd=tmp_path, capture_output=True)
    (tmp_path / "app.py").write_text("content")
    _commit(tmp_path, "test commit")

    history = get_file_history(str(tmp_path), "app.py")
    assert history is not None
    assert len(history.commits) >= 1
    assert history.commits[0].author == "Joe | Dev"
    assert history.commits[0].message == "test commit"
