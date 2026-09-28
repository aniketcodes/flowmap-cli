"""Unit tests for the qwen_ollama reranker backend — Ollama is mocked."""

import math
from unittest.mock import MagicMock, patch

import pytest
import yaml

from flowmap.config import load_config
from flowmap.search import qwen_ollama_reranker as qo
from flowmap.search.hybrid import HybridResult, _rerank_with_backend
from flowmap.search.qwen_ollama_reranker import rerank_qwen_ollama, score_from_logprobs

MODEL = "dengcao/Qwen3-Reranker-0.6B:Q8_0"


def _res(text, rrf, symbol=""):
    return HybridResult(
        repo="r", file="f.js", start_line=1, end_line=5, text=text, score=rrf, rrf_score=rrf,
        sources=["semantic"], symbol_name=symbol, chunk_type="function", signature="",
    )


def _resp(top):
    m = MagicMock()
    m.json.return_value = {"response": top[0]["token"] if top else "", "logprobs": [{"token": "x", "logprob": -1.0, "top_logprobs": top}]}
    m.raise_for_status.return_value = None
    return m


# ---------------------------------------------------------------------------
# score_from_logprobs
# ---------------------------------------------------------------------------

def test_score_is_two_way_softmax_of_yes_and_no():
    top = [{"token": "yes", "logprob": -0.1}, {"token": "no", "logprob": -2.4}, {"token": "maybe", "logprob": -9.0}]
    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.4))
    assert score_from_logprobs(top) == pytest.approx(expected)


def test_score_accepts_capitalised_and_space_prefixed_tokens():
    assert score_from_logprobs([{"token": " Yes", "logprob": -0.2}, {"token": " No", "logprob": -3.0}]) > 0.9


def test_missing_no_token_counts_as_far_below_the_list():
    top = [{"token": "yes", "logprob": -0.05}, {"token": "the", "logprob": -6.0}]
    assert score_from_logprobs(top) > 0.999


def test_missing_yes_token_scores_near_zero():
    top = [{"token": "no", "logprob": -0.05}, {"token": "the", "logprob": -6.0}]
    assert score_from_logprobs(top) < 0.001


def test_neither_token_or_empty_list_is_no_answer():
    assert score_from_logprobs([{"token": "hello", "logprob": -0.1}]) is None
    assert score_from_logprobs([]) is None


# ---------------------------------------------------------------------------
# rerank_qwen_ollama (HTTP mocked)
# ---------------------------------------------------------------------------

def _yes(lp_yes, lp_no):
    return [{"token": "yes", "logprob": lp_yes}, {"token": "no", "logprob": lp_no}]


def test_rerank_sorts_by_score_and_sends_raw_prompt():
    responses = [_resp(_yes(-3.0, -0.1)), _resp(_yes(-0.1, -3.0)), _resp(_yes(-1.0, -1.0))]
    with patch("requests.Session.post", side_effect=responses) as post:
        cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b"), _res("c", 0.7, "c")]
        out = rerank_qwen_ollama("q", cands, MODEL, "http://ollama:11434")
    assert [c.symbol_name for c in out] == ["b", "c", "a"]
    assert out[0].rerank_score > 0.9 and out[1].rerank_score == pytest.approx(0.5)
    assert [c.rrf_score for c in out] == [0.8, 0.7, 0.9]
    body = post.call_args_list[0].kwargs["json"]
    assert body["model"] == MODEL and body["raw"] is True and body["logprobs"] is True
    assert body["options"]["num_predict"] == 1 and body["options"]["temperature"] == 0
    assert "<Query>: q" in body["prompt"]
    assert "<Document>: " in body["prompt"]
    assert post.call_args_list[0].args[0] == "http://ollama:11434/api/generate"


def test_documents_are_capped_in_characters():
    with patch("requests.Session.post", return_value=_resp(_yes(-0.1, -3.0))) as post:
        rerank_qwen_ollama("q", [_res("x" * 10_000, 0.5)], MODEL)
    prompt = post.call_args.kwargs["json"]["prompt"]
    assert prompt.count("x") == qo.MAX_DOC_CHARS


def test_ties_keep_incoming_order():
    with patch("requests.Session.post", return_value=_resp(_yes(-1.0, -1.0))):
        out = rerank_qwen_ollama("q", [_res("a", 0.9, "a"), _res("b", 0.8, "b")], MODEL)
    assert [c.symbol_name for c in out] == ["a", "b"]


def test_http_failure_returns_candidates_unchanged():
    import requests
    with patch("requests.Session.post", side_effect=requests.ConnectionError("down")):
        cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b")]
        out = rerank_qwen_ollama("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["a", "b"]
    assert all(c.rerank_score == 0.0 and c.score == c.rrf_score for c in out)


def test_model_not_pulled_gives_actionable_warning(caplog):
    m = MagicMock()
    m.status_code = 404
    with patch("requests.Session.post", return_value=m), caplog.at_level("WARNING"):
        cands = [_res("a", 0.9, "a")]
        assert rerank_qwen_ollama("q", cands, MODEL) == cands
    assert f"ollama pull {MODEL}" in caplog.text


def test_missing_logprobs_in_response_degrades():
    m = MagicMock()
    m.json.return_value = {"response": "yes"}  # old Ollama: no logprobs field
    m.raise_for_status.return_value = None
    with patch("requests.Session.post", return_value=m):
        cands = [_res("a", 0.9, "a")]
        assert rerank_qwen_ollama("q", cands, MODEL) == cands


def test_empty_is_empty():
    assert rerank_qwen_ollama("q", [], MODEL) == []


# ---------------------------------------------------------------------------
# dispatch + config
# ---------------------------------------------------------------------------

def test_dispatch_qwen_ollama_passes_url():
    cands = [_res("a", 0.5)]
    with patch("flowmap.search.qwen_ollama_reranker.rerank_qwen_ollama", return_value=["ok"]) as rq:
        assert _rerank_with_backend("qwen_ollama", "q", cands, MODEL, "http://h:1") == ["ok"]
        rq.assert_called_once_with("q", cands, MODEL, "http://h:1")


def test_config_parses_qwen_ollama_block(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"repos": [], "reranking": {"backend": "qwen_ollama", "model": MODEL, "ollama_url": "http://h:1"}}))
    cfg = load_config(p)
    assert (cfg.reranking.backend, cfg.reranking.model, cfg.reranking.ollama_url) == ("qwen_ollama", MODEL, "http://h:1")
    p.write_text(yaml.safe_dump({"repos": []}))
    assert load_config(p).reranking.ollama_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# silent-failure guards + deadline
# ---------------------------------------------------------------------------

def test_no_answer_for_any_candidate_warns_and_keeps_fusion_order(caplog):
    """A broken GGUF returns a flat distribution with no yes/no tokens; that must
    not look like a successful rerank."""
    flat = [{"token": "&", "logprob": -11.93}, {"token": "*", "logprob": -11.93}]
    with patch("requests.Session.post", return_value=_resp(flat)), caplog.at_level("WARNING"):
        cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b")]
        out = rerank_qwen_ollama("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["a", "b"]
    assert all(c.rerank_score == 0.0 and c.score == c.rrf_score for c in out)
    assert "no yes/no answer for any" in caplog.text and "ollama pull" in caplog.text


def test_partial_no_answer_scores_zero_but_still_reranks(caplog):
    flat = [{"token": "&", "logprob": -11.93}]
    responses = [_resp(flat), _resp(_yes(-0.1, -3.0))]
    with patch("requests.Session.post", side_effect=responses), caplog.at_level("WARNING"):
        out = rerank_qwen_ollama("q", [_res("a", 0.9, "a"), _res("b", 0.8, "b")], MODEL)
    assert [c.symbol_name for c in out] == ["b", "a"]
    assert out[1].rerank_score == 0.0
    assert "1 of 2 candidates got no yes/no answer" in caplog.text


def test_single_request_error_does_not_sink_the_batch(caplog):
    import requests
    responses = [_resp(_yes(-0.1, -3.0)), requests.ConnectionError("blip"), _resp(_yes(-2.0, -0.2))]
    with patch("requests.Session.post", side_effect=responses), caplog.at_level("WARNING"):
        cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b"), _res("c", 0.7, "c")]
        out = rerank_qwen_ollama("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["a", "c", "b"], "a and c reranked; b unanswered sinks"
    assert out[2].symbol_name == "b" and out[2].rerank_score == 0.0
    assert "1 of 3 requests failed" in caplog.text and "blip" in caplog.text
    assert "1 of 3 candidates got no yes/no answer" in caplog.text


def test_all_requests_failing_still_degrades_with_the_error(caplog):
    import requests
    with patch("requests.Session.post", side_effect=requests.ConnectionError("down")), caplog.at_level("WARNING"):
        cands = [_res("a", 0.9, "a"), _res("b", 0.8, "b")]
        out = rerank_qwen_ollama("q", cands, MODEL)
    assert [c.symbol_name for c in out] == ["a", "b"]
    assert "reranking failed" in caplog.text and "down" in caplog.text


def test_deadline_cancels_and_returns_fusion_order(caplog, monkeypatch):
    import time as _t

    def slow_post(*a, **k):
        _t.sleep(0.3)
        return _resp(_yes(-0.1, -3.0))

    monkeypatch.setattr(qo, "DEADLINE_S", 0.2)
    with patch("requests.Session.post", side_effect=slow_post), caplog.at_level("WARNING"):
        t0 = _t.monotonic()
        cands = [_res(f"d{i}", 1 - i / 100, f"s{i}") for i in range(12)]
        out = rerank_qwen_ollama("q", cands, MODEL)
        elapsed = _t.monotonic() - t0
    assert [c.symbol_name for c in out] == [f"s{i}" for i in range(12)], "fusion order kept"
    assert all(c.rerank_score == 0.0 for c in out)
    assert "timed out" in caplog.text
    assert elapsed < 2.0, f"deadline not enforced: {elapsed:.1f}s"


def test_was_reranked_signal():
    from flowmap.search.hybrid import was_reranked
    assert was_reranked([_res("a", 0.5)]) is False
    r = _res("a", 0.5)
    r.rerank_score = 0.7
    assert was_reranked([r, _res("b", 0.4)]) is True


def test_json_envelope_carries_reranked_flag_only_when_requested():
    import json
    from flowmap.render import render_hybrid_results
    rows = [_res("a", 0.5, "a")]
    assert "reranked" not in json.loads(render_hybrid_results(rows, "q", "json"))
    assert json.loads(render_hybrid_results(rows, "q", "json", reranked=False))["reranked"] is False
    assert json.loads(render_hybrid_results([], "q", "json", reranked=False))["reranked"] is False
    rows[0].rerank_score = 0.9
    assert json.loads(render_hybrid_results(rows, "q", "json", reranked=True))["reranked"] is True
