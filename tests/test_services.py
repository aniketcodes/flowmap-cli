"""Tests for the service layer — file_resolver, symbol_lookup, indexing."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from flowmap.config import RepoConfig
from flowmap.services.file_resolver import ResolvedFile, resolve_file
from flowmap.services.symbol_lookup import SymbolMatch, resolve_symbol, get_symbol_suggestions
from flowmap.services.indexing import IndexResult, embed_chunks
from flowmap.store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(name: str, path: str) -> RepoConfig:
    return RepoConfig(name=name, path=path)


def _make_search_result(
    repo="r", file="src/app.py", start_line=10, end_line=20,
    symbol="foo", parent_symbol="", score=1.0,
):
    return SearchResult(
        repo=repo, file=file, start_line=start_line, end_line=end_line,
        text="def foo(): pass", symbol_name=symbol, chunk_type="function",
        signature="def foo():", parent_symbol=parent_symbol,
        parent_signature="", language="python", score=score,
    )


# ---------------------------------------------------------------------------
# file_resolver tests
# ---------------------------------------------------------------------------

class TestResolveFile:

    def test_explicit_repo(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        repos = [_make_repo("my-repo", str(repo_dir))]

        result = resolve_file("src/app.py", repos, explicit_repo="my-repo")
        assert result.repo_cfg.name == "my-repo"
        assert result.abs_file == repo_dir / "src/app.py"
        assert result.rel_file == "src/app.py"

    def test_explicit_repo_not_found(self, tmp_path):
        repos = [_make_repo("my-repo", str(tmp_path))]
        with pytest.raises(ValueError, match="not found in config"):
            resolve_file("src/app.py", repos, explicit_repo="nonexistent")

    def test_auto_detect_by_containing_path(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "app.py").write_text("hello")
        repos = [_make_repo("my-repo", str(repo_dir))]

        result = resolve_file(str(repo_dir / "src" / "app.py"), repos)
        assert result.repo_cfg.name == "my-repo"
        assert result.rel_file == "src/app.py"

    def test_repo_slash_path_format(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        repos = [_make_repo("my-repo", str(repo_dir))]

        result = resolve_file("my-repo/src/app.py", repos)
        assert result.repo_cfg.name == "my-repo"
        assert result.rel_file == "src/app.py"

    def test_cannot_determine_repo(self, tmp_path):
        repos = [_make_repo("my-repo", str(tmp_path / "my-repo"))]
        with pytest.raises(ValueError, match="Cannot determine repo"):
            resolve_file("/some/other/path/file.py", repos)

    def test_posix_path_normalization(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        repos = [_make_repo("my-repo", str(repo_dir))]

        result = resolve_file("src/app.py", repos, explicit_repo="my-repo")
        # rel_file should always use forward slashes
        assert "\\" not in result.rel_file


# ---------------------------------------------------------------------------
# symbol_lookup tests
# ---------------------------------------------------------------------------

class TestResolveSymbol:

    def test_file_scoped_exact(self):
        store = MagicMock()
        store.search_symbol.return_value = [
            _make_search_result(symbol="process_order"),
        ]

        match = resolve_symbol("process_order", "r", "src/app.py", store)
        assert match is not None
        assert match.source == "file_scoped"
        assert match.result.symbol_name == "process_order"

    def test_dotted_name_with_parent(self):
        store = MagicMock()
        # First call (dotted): returns method with matching parent
        store.search_symbol.return_value = [
            _make_search_result(symbol="execute", parent_symbol="OrderProcessor"),
        ]

        match = resolve_symbol("OrderProcessor.execute", "r", "src/app.py", store)
        assert match is not None
        assert match.source == "file_scoped_dotted"

    def test_dotted_name_parent_mismatch(self):
        store = MagicMock()
        # Dotted call: returns method with wrong parent → fallback to file-scoped
        wrong_parent = _make_search_result(symbol="execute", parent_symbol="WrongClass")
        store.search_symbol.side_effect = [
            [wrong_parent],   # dotted search
            [],               # file-scoped search
            [],               # global fallback
        ]

        match = resolve_symbol("OrderProcessor.execute", "r", "src/app.py", store)
        assert match is None

    def test_global_fallback_with_file_match(self):
        store = MagicMock()
        store.search_symbol.side_effect = [
            [],  # file-scoped returns nothing
            [    # global returns match from correct file
                _make_search_result(symbol="handler", file="src/app.py"),
                _make_search_result(symbol="handler", file="src/other.py"),
            ],
        ]

        match = resolve_symbol("handler", "r", "src/app.py", store)
        assert match is not None
        assert match.source == "global_fallback"
        assert match.result.file == "src/app.py"

    def test_global_fallback_suffix_path_match(self):
        store = MagicMock()
        store.search_symbol.side_effect = [
            [],  # file-scoped empty
            [_make_search_result(symbol="handler", file="packages/core/src/app.py")],  # global
        ]

        match = resolve_symbol("handler", "r", "src/app.py", store)
        assert match is not None
        assert match.source == "global_fallback"

    def test_not_found(self):
        store = MagicMock()
        store.search_symbol.return_value = []

        match = resolve_symbol("nonexistent", "r", "src/app.py", store)
        assert match is None


class TestGetSymbolSuggestions:

    def test_returns_file_symbols(self):
        store = MagicMock()
        store.get_symbols.return_value = [
            {"symbol_name": "foo", "file": "src/app.py"},
            {"symbol_name": "bar", "file": "src/app.py"},
            {"symbol_name": "baz", "file": "src/other.py"},
        ]

        suggestions = get_symbol_suggestions("r", "src/app.py", store)
        assert suggestions == ["foo", "bar"]

    def test_empty_when_no_symbols(self):
        store = MagicMock()
        store.get_symbols.return_value = []

        suggestions = get_symbol_suggestions("r", "src/app.py", store)
        assert suggestions == []


# ---------------------------------------------------------------------------
# indexing tests
# ---------------------------------------------------------------------------

class TestEmbedChunks:

    def test_batching(self):
        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        backend = MagicMock()
        backend.embed_documents.return_value = [[0.1] * 3]  # 1 embedding per call for simplicity

        # With batch_size=3, should make 4 calls (3+3+3+1)
        backend.embed_documents.side_effect = [
            [[0.1]] * 3,  # batch 0
            [[0.2]] * 3,  # batch 1
            [[0.3]] * 3,  # batch 2
            [[0.4]] * 1,  # batch 3
        ]

        embeddings = embed_chunks(chunks, backend, batch_size=3)
        assert len(embeddings) == 10
        assert backend.embed_documents.call_count == 4

    def test_progress_callback(self):
        chunks = [{"text": f"chunk {i}"} for i in range(6)]
        backend = MagicMock()
        backend.embed_documents.side_effect = [
            [[0.1]] * 3,
            [[0.2]] * 3,
        ]

        progress_calls = []
        embed_chunks(chunks, backend, batch_size=3,
                     on_progress=lambda idx, total: progress_calls.append((idx, total)))
        assert len(progress_calls) == 2
        assert progress_calls[0] == (0, 2)
        assert progress_calls[1] == (1, 2)

    def test_empty_chunks(self):
        backend = MagicMock()
        embeddings = embed_chunks([], backend)
        assert embeddings == []
        backend.embed_documents.assert_not_called()


class TestIndexResult:

    def test_fields(self):
        r = IndexResult(repo_name="test", mode="full", chunks=42, message="done")
        assert r.repo_name == "test"
        assert r.mode == "full"
        assert r.chunks == 42
