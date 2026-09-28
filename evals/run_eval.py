"""Golden retrieval eval driver.

Materialises a pinned corpus, indexes it into an isolated FlowMap data dir, runs
every golden query through the real CLI (`flowmap search --format json`), scores
the results with evals/scoring.py, and writes a JSON report.

    python -m evals.run_eval --corpus evals/corpus.smoke.yaml --golden evals/golden.smoke.yaml
    python -m evals.run_eval --corpus evals/corpus.local.yaml --golden evals/golden.local.yaml \
        --baseline evals/baseline.local.json --tolerance 0.02

The CLI is a subprocess in production (`subprocess_cli`) and an in-process
callable in tests; both return a CliResult.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import yaml

from evals.scoring import (
    GoldQuery,
    ResultRow,
    SymbolSpans,
    aggregate,
    feedback_text,
    load_golden,
    score_query,
    symbol_matches,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "evals" / "corpus.yaml"
DEFAULT_GOLDEN = ROOT / "evals" / "golden.yaml"
STAMP_NAME = ".corpus_hash"

# stderr lines from the CLI that mean "these results are not what you think".
# For the eval, a *partially* unanswered or partially failed rerank is also fatal:
# scoring it would report a degraded reranker as a healthy one.
FATAL_STDERR = ("is not indexed", "falling back to keyword-only", "was indexed with",
                "reranking failed", "reranking timed out", "no yes/no answer for any",
                "got no yes/no answer", "requests failed")


class CorpusError(RuntimeError):
    pass


class EvalError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# CLI abstraction
# ---------------------------------------------------------------------------

@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str


Cli = Callable[[list[str]], CliResult]


def inprocess_cli(args: list[str]) -> CliResult:
    """Drive the CLI in-process via click's runner. Same code path as the
    subprocess minus process startup, so models (embedding client, reranker)
    load once per run instead of once per query. Use for reranker experiments;
    the regression gate keeps the subprocess for fidelity."""
    from click.testing import CliRunner
    from flowmap.cli import main
    r = CliRunner().invoke(main, args)
    return CliResult(code=r.exit_code, stdout=r.stdout, stderr=r.stderr)


def subprocess_cli(args: list[str]) -> CliResult:
    """Run the checked-out flowmap (not a globally installed one) as a subprocess."""
    p = subprocess.run(
        [sys.executable, "-m", "flowmap", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    return CliResult(code=p.returncode, stdout=p.stdout, stderr=p.stderr)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass
class RepoSpec:
    name: str
    source: str
    commit: str
    exclude: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    name: str
    path: Path
    embedding: dict
    repos: list[RepoSpec]
    reranking: dict = field(default_factory=dict)  # optional {backend, model}; used only with --rerank

    @property
    def repo_names(self) -> set[str]:
        return {r.name for r in self.repos}


def load_corpus(path: str | Path) -> Corpus:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    def _source(s: str) -> str:
        # Git URLs and absolute/home paths pass through; relative paths resolve
        # against the manifest's directory so a tracked manifest works anywhere.
        if "://" in s or s.startswith("git@"):
            return s
        p = Path(s).expanduser()
        return str(p if p.is_absolute() else (path.parent / p).resolve())

    repos = [
        RepoSpec(name=r["name"], source=_source(str(r["source"])), commit=str(r["commit"]), exclude=list(r.get("exclude") or []))
        for r in raw.get("repos") or []
    ]
    if not repos:
        raise CorpusError(f"{path}: no repos in manifest")
    return Corpus(
        name=path.stem, path=path, embedding=dict(raw.get("embedding_profile") or {}), repos=repos,
        reranking={k: v for k, v in (raw.get("reranking") or {}).items() if k in ("backend", "model", "ollama_url")},
    )


def manifest_hash(corpus: Corpus) -> str:
    payload = {
        "embedding": corpus.embedding,
        "repos": [{"name": r.name, "commit": r.commit, "exclude": sorted(r.exclude)} for r in corpus.repos],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _git(cwd: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, p.stdout, p.stderr)
    return p.stdout.strip()


def _at_commit(clone: Path, commit: str) -> bool:
    try:
        return _git(clone, "rev-parse", "HEAD") == _git(clone, "rev-parse", "--verify", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        return False


def materialize_corpus(corpus: Corpus, work_dir: Path) -> dict[str, Path]:
    """Clone every repo at its pinned commit under work_dir/repos/<name>.

    Reuses an existing clone that is already at the pinned commit; otherwise
    re-clones. Local sources are cloned with --shared so this is instant.
    """
    repos_dir = work_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for spec in corpus.repos:
        dest = repos_dir / spec.name
        if dest.exists() and not _at_commit(dest, spec.commit):
            shutil.rmtree(dest)
        if not dest.exists():
            clone_args = ["clone", "--quiet"]
            if Path(spec.source).expanduser().is_dir():
                clone_args.append("--shared")
            try:
                _git(repos_dir, *clone_args, str(Path(spec.source).expanduser()), str(dest))
                _git(dest, "checkout", "--quiet", "--detach", spec.commit)
            except subprocess.CalledProcessError as e:
                shutil.rmtree(dest, ignore_errors=True)
                raise CorpusError(f"{spec.name}: cannot check out {spec.commit} from {spec.source}: {e.stderr.strip()}") from e
            if not _at_commit(dest, spec.commit):
                raise CorpusError(f"{spec.name}: HEAD does not match pinned commit {spec.commit}")

        ignore = dest / ".flowmapignore"
        if spec.exclude:
            ignore.write_text("\n".join(spec.exclude) + "\n")
        elif ignore.exists():
            ignore.unlink()
        out[spec.name] = dest
    return out


def write_config(corpus: Corpus, work_dir: Path, repo_paths: dict[str, Path]) -> Path:
    cfg = {
        "repos": [{"name": r.name, "path": str(repo_paths[r.name])} for r in corpus.repos],
        "data_dir": str(work_dir / "data"),
        "embedding": {k: v for k, v in corpus.embedding.items() if k in ("backend", "model", "ollama_url")},
        "reranking": {"enabled": False, **corpus.reranking},
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def ensure_index(cli: Cli, config_path: Path, work_dir: Path, corpus_hash: str) -> bool:
    """Full-index when the stamped corpus hash is missing or stale. Returns True if it indexed."""
    stamp = work_dir / "data" / STAMP_NAME
    if stamp.exists() and stamp.read_text().strip() == corpus_hash:
        return False
    r = cli(["--config", str(config_path), "index", "--full"])
    if r.code != 0:
        raise EvalError(f"index --full failed (exit {r.code}): {r.stderr.strip() or r.stdout.strip()}")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(corpus_hash + "\n")
    return True


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _parse_json_stdout(stdout: str, what: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise EvalError(f"{what}: no JSON on stdout: {stdout[:200]!r}")
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError as e:
        raise EvalError(f"{what}: bad JSON on stdout: {e}") from e


def _check_stderr(stderr: str, what: str) -> None:
    for needle in FATAL_STDERR:
        if needle in stderr:
            raise EvalError(f"{what}: CLI warned: {stderr.strip()}")


def run_query(cli: Cli, config_path: Path, q: GoldQuery, limit: int = 10, rerank: bool = False) -> tuple[list[ResultRow], str, float]:
    args = ["--config", str(config_path), "search", q.query, "--format", "json", "--limit", str(limit)]
    if q.repo:
        args += ["--repo", q.repo]
    if rerank or q.rerank:
        args.append("--rerank")
    t0 = time.perf_counter()
    r = cli(args)
    latency_ms = (time.perf_counter() - t0) * 1000
    if r.code != 0:
        raise EvalError(f"{q.id}: search failed (exit {r.code}): {r.stderr.strip() or r.stdout.strip()}")
    _check_stderr(r.stderr, q.id)
    payload = _parse_json_stdout(r.stdout, q.id)
    if (rerank or q.rerank) and payload.get("results") and payload.get("reranked") is not True:
        raise EvalError(f"{q.id}: --rerank was requested but the CLI reports reranked={payload.get('reranked')!r}; "
                        f"refusing to score fusion order as reranked. stderr: {r.stderr.strip()[:200]}")
    rows = [ResultRow.from_json(d) for d in payload.get("results") or []]
    return rows, r.stderr, latency_ms


def resolve_spans(cli: Cli, config_path: Path, queries: list[GoldQuery]) -> tuple[SymbolSpans, list[tuple[str, str, str]]]:
    """Look up each gold symbol's line span via `search --mode symbol`.

    Returns (spans, not_extracted). A gold symbol that the index does not know
    at all lands in not_extracted: that is a chunker/extractor gap, not a
    ranking miss, and is triaged separately.
    """
    wanted = sorted({(t.repo, t.file, t.symbol) for q in queries for t in q.gold if t.symbol is not None})
    spans: SymbolSpans = {}
    not_extracted: list[tuple[str, str, str]] = []
    for repo, file, symbol in wanted:
        r = cli(["--config", str(config_path), "search", symbol, "--mode", "symbol",
                 "--repo", repo, "--format", "json", "--limit", "50"])
        if r.code != 0:
            raise EvalError(f"symbol lookup {repo}/{file}::{symbol} failed: {r.stderr.strip()}")
        payload = _parse_json_stdout(r.stdout, f"symbols {symbol}")
        hit = next(
            (d for d in payload.get("results") or []
             if d.get("repo") == repo and ResultRow.from_json(d).file == file
             and symbol_matches(d.get("symbol_name") or "", symbol)),
            None,
        )
        if hit is None:
            not_extracted.append((repo, file, symbol))
        else:
            spans[(repo, file, symbol)] = (int(hit["start_line"]), int(hit["end_line"]))
    return spans, not_extracted


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _effective_reranker(config_path: Path) -> dict:
    """What the CLI will actually use for --rerank, resolved through FlowMap's own
    config loader (defaults and legacy inference included)."""
    from flowmap.config import load_config
    rc = load_config(config_path).reranking
    return {"backend": rc.backend, "model": rc.model, "ollama_url": rc.ollama_url}


def _flowmap_sha() -> str:
    try:
        return _git(ROOT, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return "unknown"


def build_report(
    cli: Cli,
    config_path: Path,
    corpus: Corpus,
    golden_path: Path,
    *,
    rerank: bool = False,
    limit: int = 10,
    only_type: str | None = None,
    only_ids: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    queries = load_golden(golden_path, corpus_repos=corpus.repo_names)
    if only_type:
        queries = [q for q in queries if q.type == only_type]
    if only_ids:
        queries = [q for q in queries if q.id in only_ids]

    spans, not_extracted = resolve_spans(cli, config_path, queries)

    scores, rows_out = [], []
    for q in queries:
        rows, stderr, latency_ms = run_query(cli, config_path, q, limit=limit, rerank=rerank)
        s = score_query(q, rows, spans, k=limit)
        scores.append(s)
        rows_out.append({
            "id": q.id, "type": q.type, "tags": list(q.tags), "query": q.query, "repo": q.repo,
            "rerank": rerank or q.rerank,
            "hit_at_1": s.hit_at_1, "mrr": round(s.mrr, 4),
            "recall_at_5": round(s.recall_at_5, 4), "recall_at_10": round(s.recall_at_10, 4),
            "first_hit_rank": s.first_hit_rank, "file_hit_rank": s.file_hit_rank,
            "sources_at_hit": s.sources_at_hit,
            "latency_ms": round(latency_ms, 1),
            "missing": [t.label for t in s.missing],
            "top": [r.label for r in rows[:limit]],
            "feedback": feedback_text(q, rows, s),
            "stderr": stderr.strip(),
        })
        if progress:
            mark = "ok " if s.recall_at_10 == 1.0 else "MISS"
            progress(f"[{mark}] {q.id} r@10={s.recall_at_10:.2f} mrr={s.mrr:.2f} {q.query[:60]}")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flowmap_sha": _flowmap_sha(),
        "corpus": {
            "name": corpus.name,
            "hash": manifest_hash(corpus),
            "repos": [{"name": r.name, "commit": r.commit} for r in corpus.repos],
        },
        "model": corpus.embedding.get("model", ""),
        "reranker": _effective_reranker(config_path) if rerank else None,
        "golden": str(golden_path),
        "flags": {"rerank": rerank, "limit": limit, "only_type": only_type, "only_ids": sorted(only_ids) if only_ids else None,
                  "in_process": cli is not subprocess_cli},
        "metrics": aggregate(scores),
        "not_extracted": [f"{r}/{f}::{s}" for r, f, s in not_extracted],
        "queries": rows_out,
    }


_GATED = ("hit_at_1", "mrr", "recall_at_10")


def diff_baseline(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """Return human-readable regressions where current < baseline - tolerance."""
    regs: list[str] = []

    def check(label: str, cur: dict, base: dict) -> None:
        for m in _GATED:
            if m in base and m in cur and cur[m] < base[m] - tolerance:
                regs.append(f"{label}.{m}: {base[m]:.3f} -> {cur[m]:.3f}")

    check("overall", current.get("overall", {}), baseline.get("overall", {}))
    for t, base in baseline.get("by_type", {}).items():
        cur = current.get("by_type", {}).get(t)
        if cur is not None:
            check(t, cur, base)
    for t, base in baseline.get("by_tag", {}).items():
        cur = current.get("by_tag", {}).get(t)
        if cur is not None:
            check(f"tag:{t}", cur, base)
    return regs


def _print_table(metrics: dict, echo: Callable[[str], None] = click.echo) -> None:
    cols = ("n", "hit_at_1", "mrr", "recall_at_5", "recall_at_10", "file_hit_at_10")
    echo(f"{'':18}" + "".join(f"{c:>16}" for c in cols))

    def row(name: str, m: dict) -> None:
        echo(f"{name:18}" + "".join(f"{m[c]:>16}" if c == "n" else f"{m[c]:>16.3f}" for c in cols))

    row("overall", metrics["overall"])
    for t, m in metrics["by_type"].items():
        row(t, m)
    for t, m in metrics.get("by_tag", {}).items():
        row(f"tag:{t}", m)
    if metrics["source_attribution"]:
        echo("first-hit sources: " + ", ".join(f"{k}={v}" for k, v in metrics["source_attribution"].items()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--corpus", "corpus_path", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_CORPUS), show_default=True)
@click.option("--golden", "golden_path", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_GOLDEN), show_default=True)
@click.option("--work-dir", type=click.Path(file_okay=False), default=None, help="Default: evals/.work/<corpus name>")
@click.option("--rerank", is_flag=True, help="Run every query with --rerank")
@click.option("--limit", default=10, show_default=True)
@click.option("--type", "only_type", type=click.Choice(["identifier", "mixed", "natural_language"]), default=None)
@click.option("--id", "only_ids", multiple=True, help="Run only these query ids (repeatable)")
@click.option("--baseline", "baseline_path", type=click.Path(dir_okay=False), default=None)
@click.option("--update-baseline", is_flag=True, help="Write this run's metrics to --baseline")
@click.option("--tolerance", default=0.02, show_default=True)
@click.option("--reindex", is_flag=True, help="Force a full re-index even if the corpus hash is unchanged")
@click.option("--in-process", is_flag=True, help="Drive the CLI in-process (one model load per run). For reranker experiments; the gate uses subprocesses.")
@click.option("--rerank-backend", type=click.Choice(["qwen_ollama", "qwen_direct"]), default=None, help="Override the manifest's reranking backend for this run")
@click.option("--rerank-model", default=None, help="Override the manifest's reranking model for this run")
@click.option("--report-dir", type=click.Path(file_okay=False), default=str(ROOT / "evals" / "reports"), show_default=True)
def main(corpus_path, golden_path, work_dir, rerank, limit, only_type, only_ids, baseline_path, update_baseline, tolerance, reindex, in_process, rerank_backend, rerank_model, report_dir):
    """Run the golden retrieval eval through the FlowMap CLI."""
    cli = inprocess_cli if in_process else subprocess_cli
    corpus = load_corpus(corpus_path)
    if rerank_backend:
        corpus.reranking["backend"] = rerank_backend
    if rerank_model:
        corpus.reranking["model"] = rerank_model
    work = Path(work_dir) if work_dir else ROOT / "evals" / ".work" / corpus.name
    work.mkdir(parents=True, exist_ok=True)

    click.echo(f"corpus {corpus.name} ({len(corpus.repos)} repos, hash {manifest_hash(corpus)})")
    paths = materialize_corpus(corpus, work)
    config_path = write_config(corpus, work, paths)

    if reindex:
        stamp = work / "data" / STAMP_NAME
        if stamp.exists():
            stamp.unlink()
    if ensure_index(subprocess_cli, config_path, work, manifest_hash(corpus)):
        click.echo("indexed corpus (full)")
    else:
        click.echo("index up to date")

    report = build_report(
        cli, config_path, corpus, Path(golden_path),
        rerank=rerank, limit=limit, only_type=only_type,
        only_ids=set(only_ids) or None, progress=click.echo,
    )

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    out = Path(report_dir) / f"{corpus.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2))

    click.echo("")
    _print_table(report["metrics"])
    if report["not_extracted"]:
        click.echo(f"not extracted ({len(report['not_extracted'])}): " + ", ".join(report["not_extracted"]))
    click.echo(f"report: {out}")

    if baseline_path:
        bp = Path(baseline_path)
        if update_baseline or not bp.exists():
            bp.write_text(json.dumps({
                "created_at": report["created_at"], "flowmap_sha": report["flowmap_sha"],
                "corpus": report["corpus"], "model": report["model"], "flags": report["flags"],
                "metrics": report["metrics"],
            }, indent=2))
            click.echo(f"baseline written: {bp}")
        else:
            baseline = json.loads(bp.read_text())
            regs = diff_baseline(report["metrics"], baseline["metrics"], tolerance)
            if regs:
                click.echo(f"REGRESSION vs {bp} (tolerance {tolerance}):", err=True)
                for r in regs:
                    click.echo(f"  {r}", err=True)
                sys.exit(2)
            click.echo(f"no regression vs {bp} (tolerance {tolerance})")


if __name__ == "__main__":
    main()
