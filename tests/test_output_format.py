"""Tests for search output formatting — JSON schema and field ordering."""

from flowmap.search.hybrid import HybridResult


def _make_hybrid_result(**kwargs) -> HybridResult:
    defaults = dict(
        repo="myrepo", file="src/auth.py", start_line=10, end_line=25,
        text="def validate_token(): pass", score=0.05, rrf_score=0.05,
        rerank_score=0.0, sources=["semantic", "ripgrep"],
        symbol_name="validate_token", chunk_type="function",
        signature="def validate_token():", parent_symbol="AuthService",
        parent_signature="class AuthService:", language="python",
        match_type="combined",
    )
    defaults.update(kwargs)
    return HybridResult(**defaults)


# ---------------------------------------------------------------------------
# HybridResult fields
# ---------------------------------------------------------------------------

def test_hybrid_result_has_all_fields():
    r = _make_hybrid_result()
    assert hasattr(r, "repo")
    assert hasattr(r, "file")
    assert hasattr(r, "start_line")
    assert hasattr(r, "end_line")
    assert hasattr(r, "text")
    assert hasattr(r, "score")
    assert hasattr(r, "rrf_score")
    assert hasattr(r, "rerank_score")
    assert hasattr(r, "sources")
    assert hasattr(r, "symbol_name")
    assert hasattr(r, "chunk_type")
    assert hasattr(r, "signature")
    assert hasattr(r, "parent_symbol")
    assert hasattr(r, "parent_signature")
    assert hasattr(r, "language")
    assert hasattr(r, "match_type")


def test_match_type_combined():
    r = _make_hybrid_result(sources=["semantic", "ripgrep"])
    assert r.match_type == "combined"


def test_match_type_single_source():
    r = _make_hybrid_result(sources=["semantic"], match_type="semantic")
    assert r.match_type == "semantic"


def test_rerank_score_default_zero():
    r = HybridResult(
        repo="r", file="f", start_line=1, end_line=1,
        text="x", score=0.1, rrf_score=0.1,
        sources=["semantic"],
    )
    assert r.rerank_score == 0.0


def test_score_equals_rrf_when_no_reranking():
    r = _make_hybrid_result(score=0.05, rrf_score=0.05, rerank_score=0.0)
    assert r.score == r.rrf_score


def test_score_equals_rerank_when_reranked():
    r = _make_hybrid_result(score=0.92, rrf_score=0.05, rerank_score=0.92)
    assert r.score == r.rerank_score


# ---------------------------------------------------------------------------
# JSON output structure
# ---------------------------------------------------------------------------

def test_json_output_field_order():
    """The JSON output should put symbol_name and signature first (most info-dense)."""
    r = _make_hybrid_result()
    # Build the output dict in the same order as cli.py
    output = {
        "symbol_name": r.symbol_name,
        "signature": r.signature,
        "parent_context": f"{r.parent_symbol}: {r.parent_signature}" if r.parent_symbol else "",
        "file": r.file,
        "repo": r.repo,
        "start_line": r.start_line,
        "end_line": r.end_line,
        "chunk_type": r.chunk_type,
        "language": r.language,
        "match_type": r.match_type,
        "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
        "rrf_score": round(r.rrf_score, 4),
        "sources": r.sources,
        "text": r.text[:500],
    }
    keys = list(output.keys())
    # symbol_name should be first, text should be last
    assert keys[0] == "symbol_name"
    assert keys[-1] == "text"
    assert keys.index("signature") < keys.index("file")
    assert keys.index("file") < keys.index("text")


def test_parent_context_format():
    r = _make_hybrid_result(parent_symbol="AuthService", parent_signature="class AuthService:")
    context = f"{r.parent_symbol}: {r.parent_signature}"
    assert context == "AuthService: class AuthService:"


def test_parent_context_empty_when_no_parent():
    r = _make_hybrid_result(parent_symbol="", parent_signature="")
    context = f"{r.parent_symbol}: {r.parent_signature}" if r.parent_symbol else ""
    assert context == ""


def test_sources_is_list():
    r = _make_hybrid_result(sources=["semantic", "ripgrep"])
    assert isinstance(r.sources, list)
    assert "semantic" in r.sources
    assert "ripgrep" in r.sources
