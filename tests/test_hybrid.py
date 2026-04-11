"""Integration-style tests for hybrid search — mock backends, degradation scenarios."""

from unittest.mock import MagicMock, patch
from flowmap.search.hybrid import hybrid_search, HybridResult, classify_query
from flowmap.store import SearchResult


def _mock_search_result(repo="r", file="f.py", start_line=10, end_line=20, symbol="foo", score=0.8):
    return SearchResult(
        repo=repo, file=file, start_line=start_line, end_line=end_line,
        text="def foo(): pass", symbol_name=symbol, chunk_type="function",
        signature="def foo():", parent_symbol="", parent_signature="",
        language="python", score=score,
    )


def _make_mock_backend():
    backend = MagicMock()
    backend.embed_query.return_value = [0.1] * 1024
    backend.dims.return_value = 1024
    return backend


def _make_mock_store(semantic_results=None, symbol_results=None, chunk_for_line=None):
    store = MagicMock()
    store.search_vector.return_value = semantic_results or []
    store.search_symbol.return_value = symbol_results or []
    store.find_chunk_containing.return_value = chunk_for_line
    # For batch ripgrep-to-chunk mapping: return the chunk in a list if provided
    if chunk_for_line:
        store.get_chunks_for_file.return_value = [chunk_for_line]
    else:
        store.get_chunks_for_file.return_value = []
    return store


# ---------------------------------------------------------------------------
# Basic hybrid search
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_hybrid_returns_semantic_results(mock_rg):
    """With no ripgrep results, should still return semantic matches."""
    backend = _make_mock_backend()
    store = _make_mock_store(semantic_results=[
        _mock_search_result(symbol="auth_handler", score=0.9),
        _mock_search_result(file="b.py", symbol="login", score=0.7),
    ])

    results = hybrid_search(
        query="how does auth work",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    assert len(results) >= 2
    assert any(r.symbol_name == "auth_handler" for r in results)


@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_hybrid_no_results(mock_rg):
    """Empty results from all sources should return empty list."""
    backend = _make_mock_backend()
    store = _make_mock_store()

    results = hybrid_search(
        query="nonexistent thing",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    assert results == []


# ---------------------------------------------------------------------------
# Source merging
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search")
def test_combined_sources_score_higher(mock_rg):
    """A result found by both ripgrep and semantic should outrank single-source results."""
    from flowmap.search.ripgrep import RgResult

    # Ripgrep finds a match at line 15 (inside the chunk at lines 10-20)
    mock_rg.return_value = [
        RgResult(repo="r", file="f.py", line=15, text="foo()"),
    ]

    chunk = _mock_search_result(start_line=10, end_line=20, symbol="foo")

    backend = _make_mock_backend()
    store = _make_mock_store(
        semantic_results=[chunk],
        chunk_for_line=chunk,  # ripgrep line 15 maps to this chunk
    )

    results = hybrid_search(
        query="foo",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    # The chunk should have both sources
    combined = [r for r in results if len(r.sources) > 1]
    assert len(combined) >= 1
    assert "ripgrep" in combined[0].sources
    assert "semantic" in combined[0].sources


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_degradation_rg_not_available(mock_rg):
    """If ripgrep returns empty, hybrid still works with semantic + symbol."""
    backend = _make_mock_backend()
    store = _make_mock_store(semantic_results=[_mock_search_result()])

    results = hybrid_search(
        query="test",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    assert len(results) >= 1


@patch("flowmap.search.hybrid.rg_search", side_effect=Exception("rg crashed"))
def test_degradation_rg_crashes(mock_rg):
    """If ripgrep crashes, hybrid should still return semantic results."""
    backend = _make_mock_backend()
    store = _make_mock_store(semantic_results=[_mock_search_result()])

    results = hybrid_search(
        query="test",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    # Should get results from semantic even though rg failed
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Query type affects results
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_identifier_query_runs_symbol_search(mock_rg):
    """Identifier queries should trigger symbol search."""
    backend = _make_mock_backend()
    store = _make_mock_store(
        symbol_results=[_mock_search_result(symbol="AuthMiddleware")],
    )

    results = hybrid_search(
        query="AuthMiddleware",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    # Symbol search should have been called
    store.search_symbol.assert_called_once()


@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_nl_query_skips_symbol_search(mock_rg):
    """Natural language queries should not run symbol search."""
    backend = _make_mock_backend()
    store = _make_mock_store(semantic_results=[_mock_search_result()])

    results = hybrid_search(
        query="how does retry work",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    # Symbol search should NOT have been called for NL query
    store.search_symbol.assert_not_called()


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search", return_value=[])
def test_limit_respected(mock_rg):
    """Should return at most `limit` results."""
    backend = _make_mock_backend()
    store = _make_mock_store(semantic_results=[
        _mock_search_result(file=f"f{i}.py", start_line=i, symbol=f"fn{i}")
        for i in range(20)
    ])

    results = hybrid_search(
        query="test query",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        reranking_enabled=False,
    )

    assert len(results) <= 5


# ---------------------------------------------------------------------------
# Regex passthrough
# ---------------------------------------------------------------------------

@patch("flowmap.search.hybrid.rg_search")
def test_hybrid_passes_regex_to_rg(mock_rg):
    """When regex=True, hybrid_search should pass it through to rg_search."""
    mock_rg.return_value = []
    backend = _make_mock_backend()
    store = _make_mock_store()

    hybrid_search(
        query="test.*pattern",
        repo_paths={"r": "/tmp/r"},
        embedding_backend=backend,
        store=store,
        limit=5,
        regex=True,
    )

    # Verify regex was passed to rg_search as keyword arg
    _, kwargs = mock_rg.call_args
    assert kwargs.get("regex") is True
