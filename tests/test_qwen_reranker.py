"""Unit tests for the qwen_direct reranker backend — no model loading."""

from unittest.mock import patch

import pytest
import yaml

from flowmap.config import load_config
from flowmap.search import qwen_reranker
from flowmap.search.hybrid import HybridResult, _rerank_with_backend
from flowmap.search.qwen_reranker import (
    DEFAULT_INSTRUCTION,
    PREFIX,
    SUFFIX,
    build_prompt,
    rerank_qwen,
    yes_probability,
)

MODEL = "Qwen/Qwen3-Reranker-0.6B"
MODEL_OLLAMA = "hf.co/Mungert/Qwen3-Reranker-0.6B-GGUF:Q8_0"


def _res(text, rrf, symbol=""):
    return HybridResult(
        repo="r", file="f.js", start_line=1, end_line=5, text=text, score=rrf, rrf_score=rrf,
        sources=["semantic"], symbol_name=symbol, chunk_type="function", signature="",
    )


class _FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, documents, instruction=DEFAULT_INSTRUCTION):
        self.calls.append((query, list(documents)))
        return list(self.scores)


@pytest.fixture(autouse=True)
def _clear_cache():
    qwen_reranker._qwen_cache.clear()
    yield
    qwen_reranker._qwen_cache.clear()


# ---------------------------------------------------------------------------
# prompt + math
# ---------------------------------------------------------------------------

def test_build_prompt_has_scaffold_instruction_query_document():
    p = build_prompt("find the lock", "const x = 1;", "judge it")
    assert p.startswith(PREFIX) and p.endswith(SUFFIX)
    assert "<Instruct>: judge it\n<Query>: find the lock\n<Document>: const x = 1;" in p


def test_build_prompt_uses_default_instruction():
    assert DEFAULT_INSTRUCTION in build_prompt("q", "d")


def test_yes_probability_is_softmax_over_two_logits():
    assert yes_probability(0.0, 0.0) == pytest.approx(0.5)
    assert yes_probability(5.0, -5.0) > 0.99
    assert yes_probability(-5.0, 5.0) < 0.01
    assert 0.0 <= yes_probability(1000.0, -1000.0) <= 1.0  # no overflow


# ---------------------------------------------------------------------------
# rerank_qwen
# ---------------------------------------------------------------------------

def test_rerank_sorts_by_score_and_sets_fields():
    qwen_reranker._qwen_cache[MODEL] = _FakeScorer([0.2, 0.9, 0.5])
    cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b"), _res("c", 0.7, "c")]
    out = rerank_qwen("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["b", "c", "a"]
    assert [c.rerank_score for c in out] == [0.9, 0.5, 0.2]
    assert [c.score for c in out] == [0.9, 0.5, 0.2]
    assert [c.rrf_score for c in out] == [0.8, 0.7, 0.9], "RRF score must survive"


def test_rerank_ties_keep_incoming_rrf_order():
    qwen_reranker._qwen_cache[MODEL] = _FakeScorer([0.5, 0.5, 0.5])
    cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b"), _res("c", 0.7, "c")]
    assert [c.symbol_name for c in rerank_qwen("q", cands, MODEL)] == ["a", "b", "c"]


def test_rerank_passes_query_and_texts_to_scorer():
    fake = _FakeScorer([0.1, 0.2])
    qwen_reranker._qwen_cache[MODEL] = fake
    rerank_qwen("the query", [_res("t1", 0.5), _res("t2", 0.4)], MODEL)
    assert fake.calls == [("the query", ["t1", "t2"])]


def test_rerank_empty_is_empty():
    assert rerank_qwen("q", [], MODEL) == []


def test_rerank_failure_returns_candidates_unchanged():
    class Boom:
        def score(self, *a, **k):
            raise RuntimeError("no gpu")

    qwen_reranker._qwen_cache[MODEL] = Boom()
    cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b")]
    out = rerank_qwen("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["a", "b"]
    assert all(c.rerank_score == 0.0 and c.score == c.rrf_score for c in out)


def test_model_load_failure_degrades_gracefully():
    with patch.object(qwen_reranker, "QwenReranker", side_effect=OSError("not cached")):
        cands = [_res("a", 0.9, "a")]
        assert rerank_qwen("q", cands, "nonexistent/model") == cands


# ---------------------------------------------------------------------------
# dispatch + config
# ---------------------------------------------------------------------------

def test_dispatch_qwen_direct_uses_qwen_backend():
    cands = [_res("a", 0.5)]
    with patch("flowmap.search.qwen_reranker.rerank_qwen", return_value=["qwen"]) as rq, \
         patch("flowmap.search.qwen_ollama_reranker.rerank_qwen_ollama", return_value=["ol"]) as ol:
        assert _rerank_with_backend("qwen_direct", "q", cands, MODEL) == ["qwen"]
        rq.assert_called_once_with("q", cands, MODEL)
        ol.assert_not_called()


def test_dispatch_unknown_backend_uses_qwen_ollama():
    cands = [_res("a", 0.5)]
    with patch("flowmap.search.qwen_ollama_reranker.rerank_qwen_ollama", return_value=["ol"]) as ol:
        assert _rerank_with_backend("bogus", "q", cands, "m", "http://h:1") == ["ol"]
        ol.assert_called_once_with("q", cands, "m", "http://h:1")


def test_cross_encoder_backend_is_gone():
    import flowmap.search.hybrid as hy
    assert not hasattr(hy, "_rerank") and not hasattr(hy, "_cross_encoder_cache")


def test_config_parses_backend(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"enabled": False, "backend": "qwen_direct", "model": MODEL}}))
    cfg = load_config(p)
    assert cfg.reranking.backend == "qwen_direct"
    assert cfg.reranking.model == MODEL
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"enabled": False}}))
    cfg = load_config(p)
    assert cfg.reranking.backend == "qwen_ollama", "Ollama-served Qwen is the shipped default since 2026-09-28"
    assert cfg.reranking.model == MODEL_OLLAMA
    assert cfg.reranking.enabled is False, "reranking stays opt-in via --rerank"


def test_legacy_cross_encoder_config_falls_back_to_default_with_warning(tmp_path, caplog):
    """A pre-2026-09-28 config naming the MS MARCO cross-encoder must not be loaded
    as a Qwen model; the removed backend maps to the default with a warning."""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"enabled": False, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}}))
    with caplog.at_level("WARNING"):
        cfg = load_config(p)
    assert (cfg.reranking.backend, cfg.reranking.model) == ("qwen_ollama", MODEL_OLLAMA)
    assert "cross-encoder" in caplog.text and "removed" in caplog.text
    # an explicit backend always wins and is left alone
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"backend": "qwen_direct", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}}))
    assert load_config(p).reranking.backend == "qwen_direct"


@pytest.mark.parametrize("model,expected", [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "qwen_ollama"),   # removed backend -> default
    ("BAAI/bge-reranker-base", "qwen_ollama"),                # any other bare HF id likewise
    ("Qwen/Qwen3-Reranker-0.6B", "qwen_direct"),              # HF id of the Qwen reranker
    ("hf.co/Mungert/Qwen3-Reranker-0.6B-GGUF:Q8_0", "qwen_ollama"),
    ("sam860/qwen3-reranker:0.6b-Q8_0", "qwen_ollama"),   # Ollama name:tag
    ("qwen3-reranker-0.6b-int8", "qwen_ollama"),          # local Ollama model
])
def test_backend_inference_from_model_name(tmp_path, model, expected):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"model": model}}))
    assert load_config(p).reranking.backend == expected


def test_empty_reranking_block_is_tolerated(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("repos: []\nreranking:\n")
    assert load_config(p).reranking.backend == "qwen_ollama"
