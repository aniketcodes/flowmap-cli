"""Unit tests for evals/scoring.py — pure matching, metrics, feedback, loading.

No Ollama, no corpus, no subprocess. These pin the semantics of the golden eval so
that the runner (phase 2) only has to worry about plumbing.
"""

import textwrap

import pytest

from evals.scoring import (
    GoldQuery,
    GoldTarget,
    ResultRow,
    aggregate,
    feedback_text,
    load_golden,
    match,
    name_overlap,
    name_words,
    score_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(repo="svc", file="src/a.ts", symbol="foo", start=10, end=20, sources=None):
    return ResultRow(
        repo=repo, file=file, symbol_name=symbol, start_line=start, end_line=end,
        sources=sources if sources is not None else ["semantic"],
    )


def _gold(repo="svc", file="src/a.ts", symbol="foo", grade="primary"):
    return GoldTarget(repo=repo, file=file, symbol=symbol, grade=grade)


def _query(gold, qid="q001", qtype="natural_language", query="find the foo thing"):
    return GoldQuery(id=qid, query=query, type=qtype, gold=list(gold))


# ---------------------------------------------------------------------------
# match()
# ---------------------------------------------------------------------------

class TestMatch:
    def test_symbol_match_hits(self):
        assert match(_row(symbol="foo"), _gold(symbol="foo"), spans={}) is True

    def test_wrong_symbol_same_file_is_not_a_match(self):
        assert match(_row(symbol="bar"), _gold(symbol="foo"), spans={}) is False

    def test_wrong_repo_never_matches(self):
        assert match(_row(repo="other"), _gold(repo="svc"), spans={}) is False

    def test_wrong_file_never_matches(self):
        assert match(_row(file="src/b.ts"), _gold(file="src/a.ts"), spans={}) is False

    def test_null_symbol_matches_any_chunk_in_file(self):
        assert match(_row(symbol="anything"), _gold(symbol=None), spans={}) is True
        assert match(_row(symbol=""), _gold(symbol=None), spans={}) is True

    def test_ripgrep_line_hit_overlapping_span_counts(self):
        spans = {("svc", "src/a.ts", "foo"): (10, 20)}
        line_hit = _row(symbol="", start=15, end=15, sources=["ripgrep"])
        assert match(line_hit, _gold(symbol="foo"), spans) is True

    def test_ripgrep_line_hit_outside_span_does_not_count(self):
        spans = {("svc", "src/a.ts", "foo"): (10, 20)}
        line_hit = _row(symbol="", start=40, end=40, sources=["ripgrep"])
        assert match(line_hit, _gold(symbol="foo"), spans) is False

    def test_ripgrep_line_hit_without_known_span_does_not_count(self):
        line_hit = _row(symbol="", start=15, end=15, sources=["ripgrep"])
        assert match(line_hit, _gold(symbol="foo"), spans={}) is False

    def test_file_path_is_normalised(self):
        assert match(_row(file="./src/a.ts"), _gold(file="src/a.ts"), spans={}) is True

    def test_gold_method_name_matches_chunkers_qualified_name(self):
        # The chunker names methods Parent.method; gold may give either form.
        assert match(_row(symbol="Greeter.hello"), _gold(symbol="hello"), spans={}) is True
        assert match(_row(symbol="Greeter.hello"), _gold(symbol="Greeter.hello"), spans={}) is True
        assert match(_row(symbol="Greeter.hello"), _gold(symbol="ello"), spans={}) is False
        assert match(_row(symbol="Other.hello"), _gold(symbol="Greeter.hello"), spans={}) is False


# ---------------------------------------------------------------------------
# score_query()
# ---------------------------------------------------------------------------

class TestScoreQuery:
    def test_hit_at_1_requires_primary(self):
        q = _query([_gold(symbol="foo", grade="acceptable"), _gold(symbol="bar", grade="primary")])
        rows = [_row(symbol="foo"), _row(symbol="bar")]
        s = score_query(q, rows, spans={})
        assert s.hit_at_1 is False
        assert s.first_hit_rank == 1  # acceptable still counts as a hit for MRR

    def test_hit_at_1_true_when_rank_1_is_primary(self):
        q = _query([_gold(symbol="foo")])
        s = score_query(q, [_row(symbol="foo")], spans={})
        assert s.hit_at_1 is True
        assert s.mrr == 1.0

    def test_mrr_uses_first_hit_of_any_grade(self):
        q = _query([_gold(symbol="foo", grade="acceptable")])
        rows = [_row(symbol="x"), _row(symbol="y"), _row(symbol="foo")]
        s = score_query(q, rows, spans={})
        assert s.first_hit_rank == 3
        assert s.mrr == pytest.approx(1 / 3)

    def test_mrr_zero_when_no_hit(self):
        q = _query([_gold(symbol="foo")])
        s = score_query(q, [_row(symbol="x")], spans={})
        assert s.mrr == 0.0
        assert s.first_hit_rank is None

    def test_recall_counts_distinct_gold_targets(self):
        q = _query([_gold(symbol="foo"), _gold(symbol="bar"), _gold(symbol="baz")])
        rows = [_row(symbol="foo"), _row(symbol="bar")]
        s = score_query(q, rows, spans={})
        assert s.recall_at_10 == pytest.approx(2 / 3)

    def test_recall_ignores_duplicate_chunks_for_same_symbol(self):
        q = _query([_gold(symbol="foo"), _gold(symbol="bar")])
        rows = [_row(symbol="foo", start=1, end=5), _row(symbol="foo", start=6, end=9)]
        s = score_query(q, rows, spans={})
        assert s.recall_at_10 == pytest.approx(0.5)

    def test_recall_at_5_only_looks_at_top_5(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(symbol=f"x{i}") for i in range(5)] + [_row(symbol="foo")]
        s = score_query(q, rows, spans={})
        assert s.recall_at_5 == 0.0
        assert s.recall_at_10 == 1.0

    def test_top_k_truncates_rows_beyond_10(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(symbol=f"x{i}") for i in range(10)] + [_row(symbol="foo")]
        s = score_query(q, rows, spans={})
        assert s.recall_at_10 == 0.0
        assert s.mrr == 0.0

    def test_file_hit_is_tracked_separately_from_hit(self):
        q = _query([_gold(symbol="foo")])
        s = score_query(q, [_row(symbol="bar")], spans={})
        assert s.recall_at_10 == 0.0
        assert s.file_hit_at_10 is True
        assert s.file_hit_rank == 1

    def test_sources_at_hit_records_first_hit_sources(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(symbol="x", sources=["ripgrep"]), _row(symbol="foo", sources=["symbol", "fts"])]
        s = score_query(q, rows, spans={})
        assert s.sources_at_hit == ["symbol", "fts"]

    def test_missing_lists_unfound_targets(self):
        foo, bar = _gold(symbol="foo"), _gold(symbol="bar")
        s = score_query(_query([foo, bar]), [_row(symbol="foo")], spans={})
        assert s.missing == [bar]

    def test_empty_results_score_zero_everywhere(self):
        s = score_query(_query([_gold()]), [], spans={})
        assert s.hit_at_1 is False
        assert s.mrr == 0.0
        assert s.recall_at_10 == 0.0
        assert s.file_hit_at_10 is False


# ---------------------------------------------------------------------------
# feedback_text()
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_feedback_lists_missing_and_top3(self):
        q = _query([_gold(symbol="foo")])
        rows = [
            _row(file="src/b.ts", symbol="one", sources=["semantic", "fts"]),
            _row(file="src/c.ts", symbol="two", sources=["ripgrep"]),
            _row(file="src/d.ts", symbol="three", sources=["symbol"]),
            _row(file="src/e.ts", symbol="four", sources=["semantic"]),
        ]
        s = score_query(q, rows, spans={})
        text = feedback_text(q, rows, s)
        assert "missing: svc/src/a.ts::foo (primary)" in text
        assert "svc/src/b.ts::one [semantic,fts]" in text
        assert "svc/src/d.ts::three [symbol]" in text
        assert "four" not in text  # only top-3 are listed
        assert "file_hit: no" in text

    def test_feedback_reports_file_hit_rank_and_symbol(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(file="src/z.ts", symbol="zzz"), _row(symbol="bar")]
        s = score_query(q, rows, spans={})
        text = feedback_text(q, rows, s)
        assert "file_hit: yes (svc/src/a.ts ranked 2 via bar)" in text

    def test_perfect_query_has_no_feedback(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(symbol="foo")]
        s = score_query(q, rows, spans={})
        assert feedback_text(q, rows, s) == ""

    def test_line_hit_without_symbol_is_shown_as_line_range(self):
        q = _query([_gold(symbol="foo")])
        rows = [_row(file="src/b.ts", symbol="", start=3, end=3, sources=["ripgrep"])]
        s = score_query(q, rows, spans={})
        assert "svc/src/b.ts:3-3 [ripgrep]" in feedback_text(q, rows, s)


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------

class TestAggregate:
    def _scores(self):
        nl = _query([_gold(symbol="foo")], qid="q1", qtype="natural_language")
        ident = _query([_gold(symbol="foo")], qid="q2", qtype="identifier", query="foo")
        mixed = _query([_gold(symbol="foo")], qid="q3", qtype="mixed", query="where is foo used")
        return [
            score_query(nl, [_row(symbol="foo", sources=["semantic"])], spans={}),          # hit@1
            score_query(ident, [_row(symbol="x"), _row(symbol="foo", sources=["symbol"])], spans={}),  # rank 2
            score_query(mixed, [_row(symbol="x")], spans={}),                                # miss
        ]

    def test_overall_means(self):
        agg = aggregate(self._scores())
        assert agg["overall"]["n"] == 3
        assert agg["overall"]["hit_at_1"] == pytest.approx(1 / 3)
        assert agg["overall"]["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
        assert agg["overall"]["recall_at_10"] == pytest.approx(2 / 3)

    def test_per_type_breakdown(self):
        agg = aggregate(self._scores())
        assert agg["by_type"]["natural_language"]["hit_at_1"] == 1.0
        assert agg["by_type"]["identifier"]["mrr"] == 0.5
        assert agg["by_type"]["mixed"]["recall_at_10"] == 0.0
        assert agg["by_type"]["identifier"]["n"] == 1

    def test_source_attribution_counts_first_hit_sources(self):
        agg = aggregate(self._scores())
        assert agg["source_attribution"] == {"semantic": 1, "symbol": 1}

    def test_empty_input(self):
        agg = aggregate([])
        assert agg["overall"]["n"] == 0
        assert agg["by_type"] == {}


# ---------------------------------------------------------------------------
# load_golden()
# ---------------------------------------------------------------------------

def _write_golden(tmp_path, body):
    p = tmp_path / "golden.yaml"
    p.write_text(textwrap.dedent(body))
    return p


VALID = """
    version: 1
    queries:
      - id: q001
        query: "how are stale search indexes rebuilt"
        type: natural_language
        gold:
          - {repo: svc, file: src/a.ts, symbol: rebuild, grade: primary}
          - {repo: svc, file: src/a.ts, symbol: stamp, grade: acceptable}
        notes: "rebuild does it; stamp records the version"
        confirmed_by: "aniket"
      - id: q002
        query: "rebuildIndex"
        type: identifier
        repo: svc
        rerank: true
        gold:
          - {repo: svc, file: src/a.ts, symbol: rebuildIndex}
"""


class TestLoader:
    def test_loads_valid_file(self, tmp_path):
        qs = load_golden(_write_golden(tmp_path, VALID))
        assert [q.id for q in qs] == ["q001", "q002"]
        assert qs[0].gold[1].grade == "acceptable"
        assert qs[0].confirmed_by == "aniket"
        assert qs[1].repo == "svc"
        assert qs[1].rerank is True
        assert qs[1].gold[0].grade == "primary"  # default grade
        assert qs[0].repo is None and qs[0].rerank is False  # defaults

    def test_type_must_match_classify_query(self, tmp_path):
        body = VALID.replace("type: identifier", "type: natural_language")
        with pytest.raises(ValueError, match="q002"):
            load_golden(_write_golden(tmp_path, body))

    def test_ids_unique(self, tmp_path):
        body = VALID.replace("id: q002", "id: q001")
        with pytest.raises(ValueError, match="q001"):
            load_golden(_write_golden(tmp_path, body))

    def test_gold_repo_must_exist_in_corpus_when_given(self, tmp_path):
        p = _write_golden(tmp_path, VALID)
        load_golden(p, corpus_repos={"svc"})  # ok
        with pytest.raises(ValueError, match="svc"):
            load_golden(p, corpus_repos={"other"})

    def test_query_needs_at_least_one_gold(self, tmp_path):
        body = VALID.replace(
            "        gold:\n          - {repo: svc, file: src/a.ts, symbol: rebuildIndex}\n",
            "        gold: []\n",
        )
        with pytest.raises(ValueError, match="q002"):
            load_golden(_write_golden(tmp_path, body))

    def test_invalid_grade_rejected(self, tmp_path):
        body = VALID.replace("grade: acceptable", "grade: maybe")
        with pytest.raises(ValueError, match="maybe"):
            load_golden(_write_golden(tmp_path, body))

    def test_unknown_type_rejected(self, tmp_path):
        body = VALID.replace("type: identifier", "type: keyword")
        with pytest.raises(ValueError, match="keyword"):
            load_golden(_write_golden(tmp_path, body))


# ---------------------------------------------------------------------------
# ResultRow.from_json — the CLI's `search --format json` row shape
# ---------------------------------------------------------------------------

def test_result_row_from_cli_json():
    d = {
        "symbol_name": "hybrid_search", "signature": "def hybrid_search(...)",
        "parent_context": "", "file": "flowmap/search/hybrid.py", "repo": "flowmap",
        "start_line": 157, "end_line": 370, "chunk_type": "function", "language": "python",
        "match_type": "combined", "rerank_score": None, "rrf_score": 0.0312,
        "sources": ["semantic", "fts"], "text": "...",
    }
    r = ResultRow.from_json(d)
    assert (r.repo, r.file, r.symbol_name) == ("flowmap", "flowmap/search/hybrid.py", "hybrid_search")
    assert (r.start_line, r.end_line) == (157, 370)
    assert r.sources == ["semantic", "fts"]


# ---------------------------------------------------------------------------
# tags + the blind guard
# ---------------------------------------------------------------------------

class TestBlindGuard:
    def test_name_words_splits_camel_snake_dot_kebab(self):
        assert name_words("calculateCancellationFeeService") == {"calculate", "cancellation", "fee", "service"}
        assert name_words("H3Util.latLngToH3") == {"h", "3", "util", "lat", "lng", "to"}
        assert name_words("fetch_user_by_id") == {"fetch", "user", "by", "id"}
        assert name_words("directions-cache.util") == {"directions", "cache", "util"}

    def test_name_overlap_counts_content_words_only(self):
        t = _gold(symbol="calculateCancellationFeeService")
        assert name_overlap("cancellation fee for a journey", t) == 2
        assert name_overlap("what does a rider pay if they back out of a booking", t) == 0
        assert name_overlap("the service", t) == 1  # 'service' is a content word; 'the' is not

    def test_file_level_gold_uses_file_stem(self):
        assert name_overlap("kafka producer plugin", _gold(symbol=None, file="src/plugins/analytics.producer.js")) == 1

    def test_loader_rejects_blind_query_that_echoes_symbol(self, tmp_path):
        body = VALID.replace(
            '        query: "how are stale search indexes rebuilt"\n        type: natural_language\n',
            '        query: "how are stale search indexes rebuilt"\n        type: natural_language\n        tags: [blind]\n',
        ).replace("symbol: rebuild, grade: primary", "symbol: rebuildStaleIndexes, grade: primary")
        with pytest.raises(ValueError, match="q001.*blind.*rebuild"):
            load_golden(_write_golden(tmp_path, body))

    def test_loader_accepts_blind_query_with_one_shared_word(self, tmp_path):
        body = VALID.replace(
            '        query: "how are stale search indexes rebuilt"\n        type: natural_language\n',
            '        query: "how are stale search indexes rebuilt"\n        type: natural_language\n        tags: [blind]\n',
        )  # gold symbol `rebuild` vs query word `rebuilt`: no exact overlap
        qs = load_golden(_write_golden(tmp_path, body))
        assert qs[0].tags == ["blind"]

    def test_guard_ignores_acceptable_targets(self, tmp_path):
        body = VALID.replace(
            '        query: "how are stale search indexes rebuilt"\n        type: natural_language\n',
            '        query: "stamp the search indexes"\n        type: natural_language\n        tags: [blind]\n',
        ).replace("symbol: stamp, grade: acceptable", "symbol: stampSearchIndexes, grade: acceptable")
        load_golden(_write_golden(tmp_path, body))  # 3 shared words, but only on the acceptable target

    def test_aggregate_by_tag(self):
        a = _query([_gold(symbol="foo")], qid="a")
        a.tags = ["blind"]
        b = _query([_gold(symbol="foo")], qid="b")
        b.tags = ["paraphrase"]
        c = _query([_gold(symbol="foo")], qid="c")
        scores = [
            score_query(a, [_row(symbol="x"), _row(symbol="foo")], spans={}),
            score_query(b, [_row(symbol="foo")], spans={}),
            score_query(c, [_row(symbol="foo")], spans={}),
        ]
        agg = aggregate(scores)
        # rank-1 row `x` is in the gold file, so it counts as a file hit
        assert agg["by_tag"]["blind"] == {"n": 1, "hit_at_1": 0.0, "mrr": 0.5, "recall_at_5": 1.0, "recall_at_10": 1.0, "file_hit_at_10": 1.0}
        assert agg["by_tag"]["paraphrase"]["n"] == 1
        assert agg["overall"]["n"] == 3 and set(agg["by_tag"]) == {"blind", "paraphrase"}
