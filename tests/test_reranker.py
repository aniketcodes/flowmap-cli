"""Tests for cross-encoder reranking logic (unit tests, no model loading)."""

from flowmap.search.hybrid import HybridResult, _rerank


def _make_result(text: str, rrf_score: float, sources: list[str] | None = None) -> HybridResult:
    return HybridResult(
        repo="r", file="f.py", start_line=1, end_line=10,
        text=text, score=rrf_score, rrf_score=rrf_score,
        sources=sources or ["semantic"],
        symbol_name="", chunk_type="function", signature="",
    )


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_rerank_without_sentence_transformers():
    """If sentence-transformers not installed, _rerank should return candidates unchanged."""
    candidates = [_make_result("hello", 0.5), _make_result("world", 0.3)]

    # _rerank will try to import CrossEncoder. If it fails (in envs without it),
    # it should return candidates as-is. If it succeeds, it will actually rerank.
    # Either way, it should not crash.
    result = _rerank("test query", candidates, "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert len(result) == len(candidates)


def test_rerank_with_empty_candidates():
    """Empty candidate list should return empty."""
    result = _rerank("query", [], "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert result == []


def test_rerank_preserves_metadata():
    """Reranking should preserve all fields except score/rerank_score."""
    candidate = HybridResult(
        repo="myrepo", file="auth.py", start_line=10, end_line=20,
        text="def validate(): pass", score=0.5, rrf_score=0.5,
        sources=["semantic", "ripgrep"], symbol_name="validate",
        chunk_type="function", signature="def validate():",
        parent_symbol="AuthService", parent_signature="class AuthService:",
        language="python", match_type="combined",
    )
    result = _rerank("auth validation", [candidate], "cross-encoder/ms-marco-MiniLM-L-6-v2")

    r = result[0]
    assert r.repo == "myrepo"
    assert r.file == "auth.py"
    assert r.symbol_name == "validate"
    assert r.sources == ["semantic", "ripgrep"]
    assert r.parent_symbol == "AuthService"
    assert r.rrf_score == 0.5  # original RRF score preserved


def test_rerank_bad_model_returns_candidates():
    """Invalid model name should fail gracefully, returning candidates unchanged."""
    candidates = [_make_result("hello", 0.5)]
    result = _rerank("query", candidates, "nonexistent/model-that-does-not-exist")
    assert len(result) == 1
    assert result[0].text == "hello"
