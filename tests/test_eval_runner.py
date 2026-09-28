"""Tests for evals/run_eval.py plumbing — corpus materialisation, config, index
stamping, CLI JSON parsing, span resolution, report and baseline diff.

No Ollama. The CLI is driven in-process through click's CliRunner with the mock
embedding backend from conftest, via the same `cli` callable the runner uses in
production (where it is a subprocess).
"""

import json
import os
import subprocess
import textwrap
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from evals import run_eval
from evals.run_eval import (
    CliResult,
    CorpusError,
    EvalError,
    build_report,
    diff_baseline,
    ensure_index,
    load_corpus,
    manifest_hash,
    materialize_corpus,
    resolve_spans,
    run_query,
    write_config,
)
from evals.scoring import GoldQuery, GoldTarget
from tests.conftest import MockBackend

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.t",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.t",
}


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=GIT_ENV, check=True).stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    """A git repo with two commits; the second adds a file the eval must NOT see."""
    repo = tmp_path / "src-repo"
    repo.mkdir()
    (repo / "math_utils.py").write_text("def add(a, b):\n    return a + b\n\n\ndef multiply(x, y):\n    return x * y\n")
    (repo / "greeter.py").write_text("class Greeter:\n    def hello(self, name):\n        return f'Hello {name}'\n")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n')
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "later.py").write_text("def later():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "two")
    return repo, first


def _corpus_dict(repo, commit, exclude=None):
    return {
        "version": 1,
        "embedding_profile": {"backend": "ollama", "model": "test:mock"},
        "repos": [{"name": "svc", "source": str(repo), "commit": commit, "exclude": exclude or []}],
    }


def _write_corpus(tmp_path, d, name="corpus.smoke.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def _inprocess_cli(args) -> CliResult:
    from flowmap.cli import main
    with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
        r = CliRunner().invoke(main, args)
    return CliResult(code=r.exit_code, stdout=r.stdout, stderr=r.stderr)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class TestCorpus:
    def test_load_corpus_reads_manifest(self, tmp_path, source_repo):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha, ["package-lock.json"])))
        assert c.name == "corpus.smoke"
        assert c.embedding["model"] == "test:mock"
        assert c.repos[0].name == "svc"
        assert c.repos[0].commit == sha
        assert c.repos[0].exclude == ["package-lock.json"]

    def test_relative_source_resolves_against_manifest_dir(self, tmp_path, source_repo):
        repo, sha = source_repo
        d = _corpus_dict(repo, sha)
        d["repos"][0]["source"] = "../src-repo"          # manifest lives in tmp_path/manifests/
        (tmp_path / "manifests").mkdir()
        c = load_corpus(_write_corpus(tmp_path / "manifests", d))
        assert c.repos[0].source == str(repo.resolve())
        d["repos"][0]["source"] = "https://example.com/x.git"
        c = load_corpus(_write_corpus(tmp_path / "manifests", d, "u.yaml"))
        assert c.repos[0].source == "https://example.com/x.git"

    def test_manifest_hash_changes_with_commit_model_or_exclude(self, tmp_path, source_repo):
        repo, sha = source_repo
        base = manifest_hash(load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha))))
        other = _corpus_dict(repo, sha)
        other["embedding_profile"]["model"] = "other"
        assert manifest_hash(load_corpus(_write_corpus(tmp_path, other, "a.yaml"))) != base
        other = _corpus_dict(repo, sha, ["x"])
        assert manifest_hash(load_corpus(_write_corpus(tmp_path, other, "b.yaml"))) != base
        other = _corpus_dict(repo, "0" * 40)
        assert manifest_hash(load_corpus(_write_corpus(tmp_path, other, "c.yaml"))) != base

    def test_materialize_local_corpus_checks_out_pinned_sha(self, tmp_path, source_repo):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha, ["package-lock.json"])))
        work = tmp_path / "work"
        paths = materialize_corpus(c, work)
        clone = paths["svc"]
        assert _git(clone, "rev-parse", "HEAD") == sha
        assert (clone / "math_utils.py").exists()
        assert not (clone / "later.py").exists(), "second commit must not be visible"
        assert (clone / ".flowmapignore").read_text().splitlines() == ["package-lock.json"]

    def test_materialize_refuses_sha_mismatch(self, tmp_path, source_repo):
        repo, _ = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, "deadbeef" * 5)))
        with pytest.raises(CorpusError, match="svc"):
            materialize_corpus(c, tmp_path / "work")

    def test_materialize_reuses_existing_clone_at_same_commit(self, tmp_path, source_repo):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha)))
        work = tmp_path / "work"
        first = materialize_corpus(c, work)["svc"]
        marker = first / ".marker"
        marker.write_text("x")
        second = materialize_corpus(c, work)["svc"]
        assert second == first and marker.exists(), "clone at the right sha must be reused, not re-cloned"

    def test_materialize_replaces_clone_at_wrong_commit(self, tmp_path, source_repo):
        repo, sha = source_repo
        second_sha = _git(repo, "rev-parse", "HEAD")
        work = tmp_path / "work"
        materialize_corpus(load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, second_sha), "a.yaml")), work)
        clone = materialize_corpus(load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha), "b.yaml")), work)["svc"]
        assert _git(clone, "rev-parse", "HEAD") == sha
        assert not (clone / "later.py").exists()


# ---------------------------------------------------------------------------
# Config + index stamp
# ---------------------------------------------------------------------------

class TestConfigAndIndex:
    def test_write_config_is_isolated(self, tmp_path, source_repo):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha)))
        work = tmp_path / "work"
        cfg_path = write_config(c, work, {"svc": work / "repos" / "svc"})
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["repos"] == [{"name": "svc", "path": str(work / "repos" / "svc")}]
        assert cfg["data_dir"] == str(work / "data")
        assert cfg["embedding"]["model"] == "test:mock"
        assert cfg["reranking"]["enabled"] is False

    def test_write_config_passes_reranker_through_but_never_enables_it(self, tmp_path, source_repo):
        repo, sha = source_repo
        d = _corpus_dict(repo, sha)
        d["reranking"] = {"backend": "qwen_direct", "model": "Qwen/Qwen3-Reranker-0.6B", "enabled": True}
        c = load_corpus(_write_corpus(tmp_path, d))
        assert c.reranking == {"backend": "qwen_direct", "model": "Qwen/Qwen3-Reranker-0.6B"}
        cfg = yaml.safe_load(write_config(c, tmp_path / "work", {"svc": tmp_path / "x"}).read_text())
        assert cfg["reranking"] == {"enabled": False, "backend": "qwen_direct", "model": "Qwen/Qwen3-Reranker-0.6B"}
        # reranker choice must not change the corpus hash (it does not affect the index)
        assert manifest_hash(c) == manifest_hash(load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha), "b.yaml")))

    def test_stale_manifest_hash_forces_full_index(self, tmp_path):
        calls: list[list[str]] = []

        def fake_cli(args):
            calls.append(args)
            return CliResult(code=0, stdout="", stderr="")

        work = tmp_path / "work"
        cfg = tmp_path / "config.yaml"
        cfg.write_text("{}")
        assert ensure_index(fake_cli, cfg, work, "h1") is True
        assert calls[-1][-2:] == ["index", "--full"] and "--config" in calls[-1]
        assert ensure_index(fake_cli, cfg, work, "h1") is False, "same hash: skip"
        assert len(calls) == 1
        assert ensure_index(fake_cli, cfg, work, "h2") is True, "new hash: re-index"
        assert len(calls) == 2

    def test_failed_index_does_not_write_stamp(self, tmp_path):
        def failing_cli(args):
            return CliResult(code=1, stdout="", stderr="boom")

        work = tmp_path / "work"
        cfg = tmp_path / "config.yaml"
        cfg.write_text("{}")
        with pytest.raises(EvalError, match="boom"):
            ensure_index(failing_cli, cfg, work, "h1")
        assert not (work / "data" / ".corpus_hash").exists()


# ---------------------------------------------------------------------------
# Queries through the CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def indexed(tmp_path, source_repo):
    """Materialised + indexed corpus using the in-process CLI. Returns config path."""
    repo, sha = source_repo
    c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha, ["package-lock.json"])))
    work = tmp_path / "work"
    paths = materialize_corpus(c, work)
    cfg_path = write_config(c, work, paths)
    ensure_index(_inprocess_cli, cfg_path, work, manifest_hash(c))
    return cfg_path


class TestRunQuery:
    def test_run_query_parses_json_rows(self, indexed):
        q = GoldQuery(id="q1", query="add", type="identifier",
                      gold=[GoldTarget(repo="svc", file="math_utils.py", symbol="add")])
        rows, stderr, latency_ms = run_query(_inprocess_cli, indexed, q)
        assert rows, "expected results for a symbol that exists"
        assert any(r.symbol_name == "add" and r.file == "math_utils.py" for r in rows)
        assert latency_ms >= 0

    def test_run_query_passes_repo_filter_and_rerank(self, tmp_path):
        seen: list[list[str]] = []

        def fake_cli(args):
            seen.append(args)
            return CliResult(code=0, stdout=json.dumps({"query": "x", "mode": "hybrid", "results": []}), stderr="")

        q = GoldQuery(id="q1", query="x", type="natural_language", repo="svc", rerank=True,
                      gold=[GoldTarget(repo="svc", file="a.py")])
        run_query(fake_cli, tmp_path / "c.yaml", q, rerank=False)
        args = seen[0]
        assert "--repo" in args and args[args.index("--repo") + 1] == "svc"
        assert "--rerank" in args
        assert "--format" in args and args[args.index("--format") + 1] == "json"

    def test_global_rerank_flag_applies_to_all_queries(self, tmp_path):
        seen: list[list[str]] = []

        def fake_cli(args):
            seen.append(args)
            return CliResult(code=0, stdout=json.dumps({"results": []}), stderr="")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        run_query(fake_cli, tmp_path / "c.yaml", q, rerank=True)
        assert "--rerank" in seen[0]

    def test_rerank_requested_but_not_reranked_fails_run(self, tmp_path):
        def fake_cli(args):
            body = {"query": "x", "mode": "hybrid", "reranked": False,
                    "results": [{"repo": "svc", "file": "a.py", "symbol_name": "f", "start_line": 1, "end_line": 2, "sources": ["semantic"]}]}
            return CliResult(code=0, stdout=json.dumps(body), stderr="")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="reranked=False"):
            run_query(fake_cli, tmp_path / "c.yaml", q, rerank=True)
        run_query(fake_cli, tmp_path / "c.yaml", q, rerank=False)  # fine when not requested

    def test_reranker_failure_on_stderr_fails_run(self, tmp_path):
        def fake_cli(args):
            return CliResult(code=0, stdout=json.dumps({"results": [], "reranked": False}),
                             stderr="WARNING: Qwen/Ollama reranking failed (m): connection refused")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="reranking failed"):
            run_query(fake_cli, tmp_path / "c.yaml", q, rerank=True)

    @pytest.mark.parametrize("line", [
        "WARNING: Qwen/Ollama reranking: 3 of 30 candidates got no yes/no answer; scored as 0",
        "WARNING: Qwen/Ollama reranking: 1 of 30 requests failed (last: boom); those candidates scored as unanswered",
    ])
    def test_partially_degraded_rerank_fails_run(self, tmp_path, line):
        def fake_cli(args):
            body = {"results": [{"repo": "svc", "file": "a.py", "symbol_name": "f", "start_line": 1, "end_line": 2, "sources": ["semantic"]}], "reranked": True}
            return CliResult(code=0, stdout=json.dumps(body), stderr=line)

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="CLI warned"):
            run_query(fake_cli, tmp_path / "c.yaml", q, rerank=True)

    def test_stderr_not_indexed_warning_fails_run(self, tmp_path):
        def fake_cli(args):
            return CliResult(code=0, stdout=json.dumps({"results": []}),
                             stderr="Warning: profile 'default' is not indexed. Run: flowmap index")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="not indexed"):
            run_query(fake_cli, tmp_path / "c.yaml", q)

    def test_embedding_fallback_warning_fails_run(self, tmp_path):
        def fake_cli(args):
            return CliResult(code=0, stdout=json.dumps({"results": []}),
                             stderr="Warning: embedding unavailable (x), falling back to keyword-only.")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="falling back"):
            run_query(fake_cli, tmp_path / "c.yaml", q)

    def test_nonzero_exit_fails_run(self, tmp_path):
        def fake_cli(args):
            return CliResult(code=1, stdout="", stderr="Error: kaboom")

        q = GoldQuery(id="q1", query="x", type="natural_language", gold=[GoldTarget(repo="svc", file="a.py")])
        with pytest.raises(EvalError, match="kaboom"):
            run_query(fake_cli, tmp_path / "c.yaml", q)


class TestResolveSpans:
    def test_resolves_spans_and_reports_not_extracted(self, indexed):
        queries = [
            GoldQuery(id="q1", query="add", type="natural_language",
                      gold=[GoldTarget(repo="svc", file="math_utils.py", symbol="add"),
                            GoldTarget(repo="svc", file="greeter.py", symbol="Greeter")]),
            GoldQuery(id="q2", query="ghost", type="natural_language",
                      gold=[GoldTarget(repo="svc", file="math_utils.py", symbol="ghost"),
                            # small class is chunked whole: its method is not a symbol
                            GoldTarget(repo="svc", file="greeter.py", symbol="hello"),
                            GoldTarget(repo="svc", file="greeter.py", symbol=None)]),
        ]
        spans, not_extracted = resolve_spans(_inprocess_cli, indexed, queries)
        assert ("svc", "math_utils.py", "add") in spans
        start, end = spans[("svc", "math_utils.py", "add")]
        assert 1 <= start <= end
        assert spans[("svc", "greeter.py", "Greeter")] == (1, 3)
        assert not_extracted == [("svc", "greeter.py", "hello"), ("svc", "math_utils.py", "ghost")]
        assert all(t[2] is not None for t in spans), "file-level gold has no span to resolve"


# ---------------------------------------------------------------------------
# Report + baseline
# ---------------------------------------------------------------------------

def _golden_file(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text(textwrap.dedent("""
        version: 1
        queries:
          - id: q001
            query: "add"
            type: natural_language
            gold: [{repo: svc, file: math_utils.py, symbol: add}]
          - id: q002
            query: "Greeter"
            type: identifier
            gold: [{repo: svc, file: greeter.py, symbol: Greeter}]
          - id: q003
            query: "ghost"
            type: natural_language
            gold: [{repo: svc, file: math_utils.py, symbol: ghost}]
    """))
    return p


class TestReport:
    def test_report_contains_provenance_and_per_query_rows(self, tmp_path, source_repo, indexed):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha)))
        report = build_report(_inprocess_cli, indexed, c, _golden_file(tmp_path))
        assert report["corpus"]["name"] == c.name
        assert report["corpus"]["hash"] == manifest_hash(c)
        assert report["corpus"]["repos"] == [{"name": "svc", "commit": sha}]
        assert report["model"] == "test:mock"
        assert len(report["flowmap_sha"]) >= 7
        assert "created_at" in report
        assert set(report["metrics"]) == {"overall", "by_type", "by_tag", "source_attribution"}
        assert all("tags" in r for r in report["queries"])
        assert report["metrics"]["overall"]["n"] == 3
        rows = {r["id"]: r for r in report["queries"]}
        assert rows["q001"]["recall_at_10"] == 1.0
        assert rows["q003"]["recall_at_10"] == 0.0
        assert rows["q003"]["feedback"].startswith("missing: svc/math_utils.py::ghost")
        assert rows["q001"]["feedback"] == ""
        assert report["not_extracted"] == ["svc/math_utils.py::ghost"]
        assert all("latency_ms" in r and "top" in r for r in report["queries"])

    def test_build_report_filters_by_type_and_id(self, tmp_path, source_repo, indexed):
        repo, sha = source_repo
        c = load_corpus(_write_corpus(tmp_path, _corpus_dict(repo, sha)))
        g = _golden_file(tmp_path)
        assert [r["id"] for r in build_report(_inprocess_cli, indexed, c, g, only_type="identifier")["queries"]] == ["q002"]
        assert [r["id"] for r in build_report(_inprocess_cli, indexed, c, g, only_ids={"q001"})["queries"]] == ["q001"]


class TestBaselineDiff:
    def _metrics(self, hit=0.5, mrr=0.6, r10=0.8, ident_r10=0.9):
        return {
            "overall": {"n": 4, "hit_at_1": hit, "mrr": mrr, "recall_at_5": 0.7, "recall_at_10": r10, "file_hit_at_10": 0.9},
            "by_type": {"identifier": {"n": 2, "hit_at_1": 1.0, "mrr": 1.0, "recall_at_5": 0.9, "recall_at_10": ident_r10, "file_hit_at_10": 1.0}},
            "source_attribution": {},
        }

    def test_no_regression_within_tolerance(self):
        assert diff_baseline(self._metrics(r10=0.79), self._metrics(), tolerance=0.02) == []

    def test_flags_overall_regression_beyond_tolerance(self):
        regs = diff_baseline(self._metrics(mrr=0.5), self._metrics(), tolerance=0.02)
        assert any("overall.mrr" in r for r in regs)

    def test_flags_per_type_regression(self):
        regs = diff_baseline(self._metrics(ident_r10=0.5), self._metrics(), tolerance=0.02)
        assert any("identifier.recall_at_10" in r for r in regs)

    def test_new_type_in_current_is_not_a_regression(self):
        cur = self._metrics()
        cur["by_type"]["mixed"] = {"n": 1, "hit_at_1": 0, "mrr": 0, "recall_at_5": 0, "recall_at_10": 0, "file_hit_at_10": 0}
        assert diff_baseline(cur, self._metrics(), tolerance=0.02) == []

    def test_improvements_are_not_regressions(self):
        assert diff_baseline(self._metrics(hit=0.9, mrr=0.9, r10=1.0), self._metrics(), tolerance=0.0) == []


# ---------------------------------------------------------------------------
# Subprocess CLI wrapper
# ---------------------------------------------------------------------------

def test_inprocess_cli_matches_subprocess_contract():
    r = run_eval.inprocess_cli(["--help"])
    assert r.code == 0 and "search" in r.stdout


def test_subprocess_cli_runs_checked_out_flowmap():
    r = run_eval.subprocess_cli(["--help"])
    assert r.code == 0
    assert "search" in r.stdout
