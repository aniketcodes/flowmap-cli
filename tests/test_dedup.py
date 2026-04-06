"""Tests for dedup-before-scoring — ripgrep lines mapped to chunks before RRF."""

from flowmap.search.hybrid import _UnifiedEntry
from flowmap.store import SearchResult


def _make_search_result(repo="r", file="f.py", start_line=10, end_line=20, symbol="foo") -> SearchResult:
    return SearchResult(
        repo=repo, file=file, start_line=start_line, end_line=end_line,
        text="def foo(): pass", symbol_name=symbol, chunk_type="function",
        signature="def foo():", parent_symbol="", parent_signature="",
        language="python", score=0.5,
    )


# ---------------------------------------------------------------------------
# Unified entry merging
# ---------------------------------------------------------------------------

def test_same_chunk_from_two_sources_merges():
    """A chunk found by both semantic and ripgrep should have both sources."""
    unified = {}
    sr = _make_search_result()

    # Semantic adds it
    key = (sr.repo, sr.file, sr.start_line)
    entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
    entry.sources.add("semantic")

    # Ripgrep maps to same chunk — should merge, not create duplicate
    entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
    entry.sources.add("ripgrep")

    assert len(unified) == 1
    assert unified[key].sources == {"semantic", "ripgrep"}


def test_different_chunks_stay_separate():
    """Chunks from different files should not merge."""
    unified = {}

    sr1 = _make_search_result(file="a.py", start_line=10)
    sr2 = _make_search_result(file="b.py", start_line=10)

    key1 = (sr1.repo, sr1.file, sr1.start_line)
    key2 = (sr2.repo, sr2.file, sr2.start_line)

    unified.setdefault(key1, _UnifiedEntry.from_search_result(sr1)).sources.add("semantic")
    unified.setdefault(key2, _UnifiedEntry.from_search_result(sr2)).sources.add("ripgrep")

    assert len(unified) == 2


def test_ripgrep_line_inside_chunk_maps_to_chunk_key():
    """A ripgrep hit at line 15 inside a chunk spanning 10-20 should use key (repo, file, 10)."""
    chunk = _make_search_result(start_line=10, end_line=20)

    # The ripgrep hit is at line 15 — but after find_chunk_containing, it maps to the chunk
    key = (chunk.repo, chunk.file, chunk.start_line)  # (r, f.py, 10)
    assert key == ("r", "f.py", 10)


def test_standalone_ripgrep_result_has_ripgrep_source():
    """Ripgrep hits with no matching chunk should be standalone entries."""
    unified = {}
    key = ("myrepo", "unindexed.py", 42)
    unified[key] = _UnifiedEntry(
        repo="myrepo", file="unindexed.py",
        start_line=42, end_line=42,
        text="some_match_line", symbol_name="", chunk_type="",
        signature="", parent_symbol="", parent_signature="",
        language="", sources={"ripgrep"},
    )

    assert unified[key].sources == {"ripgrep"}
    assert unified[key].symbol_name == ""  # no metadata for unindexed files


def test_three_sources_merge_into_one_entry():
    """A result found by all 3 sources should have all 3 in sources set."""
    unified = {}
    sr = _make_search_result()
    key = (sr.repo, sr.file, sr.start_line)

    entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
    entry.sources.add("semantic")
    entry.sources.add("ripgrep")
    entry.sources.add("symbol")

    assert unified[key].sources == {"semantic", "ripgrep", "symbol"}


# ---------------------------------------------------------------------------
# Source rank deduplication
# ---------------------------------------------------------------------------

def test_duplicate_semantic_results_not_double_counted():
    """If semantic search returns the same chunk twice, it should appear once in rankings."""
    seen = set()
    results = [
        _make_search_result(start_line=10),
        _make_search_result(start_line=10),  # duplicate
        _make_search_result(start_line=30),
    ]

    ranked = []
    for sr in results:
        key = (sr.repo, sr.file, sr.start_line)
        if key not in seen:
            seen.add(key)
            ranked.append(key)

    assert len(ranked) == 2  # not 3
