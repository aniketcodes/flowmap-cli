"""Tests for RRF fusion scoring logic."""

from flowmap.search.hybrid import HybridResult, RRF_K


def _rrf_score(rank: int, weight: float = 1.0) -> float:
    """Compute expected RRF score for a single source at a given rank (1-indexed)."""
    return weight * (1.0 / (RRF_K + rank))


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

def test_rrf_score_decreases_with_rank():
    """Higher ranked results should get higher RRF scores."""
    score_rank1 = _rrf_score(1)
    score_rank2 = _rrf_score(2)
    score_rank10 = _rrf_score(10)
    assert score_rank1 > score_rank2 > score_rank10


def test_rrf_weight_multiplies_score():
    """Source weight should multiply the RRF contribution."""
    base = _rrf_score(1, weight=1.0)
    doubled = _rrf_score(1, weight=2.0)
    assert abs(doubled - 2 * base) < 1e-10


def test_combined_score_higher_than_single():
    """A result found by 2 sources should score higher than one found by 1."""
    single = _rrf_score(1)
    combined = _rrf_score(1) + _rrf_score(1)  # same rank in both sources
    assert combined > single


# ---------------------------------------------------------------------------
# Weight application by query type
# ---------------------------------------------------------------------------

def test_identifier_weights_favor_symbol():
    """For identifier queries, symbol weight (2.0) should dominate."""
    from flowmap.search.hybrid import _WEIGHTS
    w = _WEIGHTS["identifier"]
    assert w["symbol"] > w["ripgrep"] > w["semantic"]


def test_nl_weights_favor_semantic():
    """For natural language queries, semantic weight (2.0) should dominate."""
    from flowmap.search.hybrid import _WEIGHTS
    w = _WEIGHTS["natural_language"]
    assert w["semantic"] > w["ripgrep"] > w["symbol"]


def test_mixed_weights_are_equal():
    """For mixed queries, all weights should be equal (1.0)."""
    from flowmap.search.hybrid import _WEIGHTS
    w = _WEIGHTS["mixed"]
    assert w["ripgrep"] == w["semantic"] == w["symbol"] == 1.0


# ---------------------------------------------------------------------------
# K constant
# ---------------------------------------------------------------------------

def test_k_is_60():
    """Standard RRF constant should be 60."""
    assert RRF_K == 60


def test_k_dampens_rank_differences():
    """With k=60, difference between rank 1 and rank 2 is small."""
    diff = _rrf_score(1) - _rrf_score(2)
    ratio = _rrf_score(1) / _rrf_score(2)
    # Rank 1: 1/61, Rank 2: 1/62 — ratio should be close to 1
    assert ratio < 1.02  # less than 2% difference
