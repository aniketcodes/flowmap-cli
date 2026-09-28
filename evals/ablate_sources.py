"""Source ablation: how much does each of the four search legs contribute?

Runs the golden set under nine weight configurations, in-process through the
CLI's `search` command with `flowmap.search.hybrid._WEIGHTS` patched so that
excluded legs get weight 0 (and rows that only those legs produced are dropped):

    all                        the shipped weights
    only:<leg>       x4        one leg alone
    without:<leg>    x4        leave-one-out -> marginal value = all - without:<leg>

This is an experiment, not the regression gate, hence in-process (one model
load, no subprocess per query). Scoring and spans are the same as run_eval.

    python -m evals.ablate_sources --corpus evals/corpus.local.yaml --golden evals/golden.local.yaml
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
from click.testing import CliRunner

import flowmap.search.hybrid as hybrid_mod
from evals.run_eval import (
    ROOT,
    CliResult,
    _check_stderr,
    _parse_json_stdout,
    ensure_index,
    load_corpus,
    manifest_hash,
    materialize_corpus,
    resolve_spans,
    subprocess_cli,
    write_config,
)
from evals.scoring import GoldQuery, ResultRow, aggregate, load_golden, score_query

LEGS = ("ripgrep", "fts", "semantic", "symbol")


def inprocess_cli(args: list[str]) -> CliResult:
    from flowmap.cli import main
    r = CliRunner().invoke(main, args)
    return CliResult(code=r.exit_code, stdout=r.stdout, stderr=r.stderr)


def _configs() -> dict[str, set[str]]:
    cfgs: dict[str, set[str]] = {"all": set(LEGS)}
    for leg in LEGS:
        cfgs[f"only:{leg}"] = {leg}
    for leg in LEGS:
        cfgs[f"without:{leg}"] = set(LEGS) - {leg}
    return cfgs


class _PatchedWeights:
    def __init__(self, active: set[str]):
        self.active = active
        self.saved = None

    def __enter__(self):
        self.saved = hybrid_mod._WEIGHTS
        hybrid_mod._WEIGHTS = {
            qt: {src: (w if src in self.active else 0.0) for src, w in ws.items()}
            for qt, ws in self.saved.items()
        }

    def __exit__(self, *exc):
        hybrid_mod._WEIGHTS = self.saved


def _search(config_path: Path, q: GoldQuery, limit: int) -> list[ResultRow]:
    args = ["--config", str(config_path), "search", q.query, "--format", "json", "--limit", str(limit)]
    if q.repo:
        args += ["--repo", q.repo]
    r = inprocess_cli(args)
    if r.code != 0:
        raise RuntimeError(f"{q.id}: search failed (exit {r.code}): {r.stderr.strip() or r.stdout.strip()}")
    _check_stderr(r.stderr, q.id)
    payload = _parse_json_stdout(r.stdout, q.id)
    # Drop rows that only zero-weight legs produced (rrf_score rounds to 0.0).
    return [ResultRow.from_json(d) for d in payload.get("results") or [] if (d.get("rrf_score") or 0) > 0]


def run_ablation(config_path: Path, golden_path: Path, corpus_repos: set[str], limit: int = 10, echo=click.echo) -> dict:
    queries = load_golden(golden_path, corpus_repos=corpus_repos)
    spans, not_extracted = resolve_spans(inprocess_cli, config_path, queries)
    results: dict[str, dict] = {}
    for name, active in _configs().items():
        echo(f"  {name:18} ", nl=False)
        with _PatchedWeights(active):
            scores = [score_query(q, _search(config_path, q, limit), spans, k=limit) for q in queries]
        results[name] = aggregate(scores)
        m = results[name]["overall"]
        echo(f"hit@1={m['hit_at_1']:.3f} mrr={m['mrr']:.3f} r@10={m['recall_at_10']:.3f}")
    return {"configs": results, "not_extracted": [f"{r}/{f}::{s}" for r, f, s in not_extracted]}


def _table(results: dict[str, dict], echo=click.echo) -> None:
    types = ["overall"] + sorted({t for r in results.values() for t in r["by_type"]})
    metrics = ("hit_at_1", "mrr", "recall_at_10")

    def cell(r: dict, t: str, m: str) -> str:
        block = r["overall"] if t == "overall" else r["by_type"].get(t)
        return f"{block[m]:.3f}" if block else "  -  "

    for m in metrics:
        echo("")
        echo(f"{m:>20}" + "".join(f"{t[:16]:>18}" for t in types))
        for name, r in results.items():
            echo(f"{name:>20}" + "".join(f"{cell(r, t, m):>18}" for t in types))

    echo("")
    echo("marginal value (all - without:<leg>), overall / natural_language:")
    base = results["all"]
    for leg in LEGS:
        w = results[f"without:{leg}"]
        line = f"  {leg:9}"
        for m in metrics:
            d_all = base["overall"][m] - w["overall"][m]
            nb, nw = base["by_type"].get("natural_language"), w["by_type"].get("natural_language")
            d_nl = (nb[m] - nw[m]) if nb and nw else 0.0
            line += f"  {m}: {d_all:+.3f} / {d_nl:+.3f}"
        echo(line)


@click.command()
@click.option("--corpus", "corpus_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--golden", "golden_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--work-dir", type=click.Path(file_okay=False), default=None)
@click.option("--limit", default=10, show_default=True)
@click.option("--report-dir", type=click.Path(file_okay=False), default=str(ROOT / "evals" / "reports"), show_default=True)
def main(corpus_path, golden_path, work_dir, limit, report_dir):
    corpus = load_corpus(corpus_path)
    work = Path(work_dir) if work_dir else ROOT / "evals" / ".work" / corpus.name
    paths = materialize_corpus(corpus, work)
    config_path = write_config(corpus, work, paths)
    ensure_index(subprocess_cli, config_path, work, manifest_hash(corpus))

    click.echo(f"ablation over {corpus.name}, weights from flowmap.search.hybrid._WEIGHTS")
    out = run_ablation(config_path, Path(golden_path), corpus.repo_names, limit=limit)
    _table(out["configs"])
    if out["not_extracted"]:
        click.echo(f"not extracted: {out['not_extracted']}")

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / f"ablation-{corpus.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps({"corpus": {"name": corpus.name, "hash": manifest_hash(corpus)}, "golden": str(golden_path),
                                "weights": hybrid_mod._WEIGHTS, **out}, indent=2))
    click.echo(f"report: {path}")


if __name__ == "__main__":
    main()
