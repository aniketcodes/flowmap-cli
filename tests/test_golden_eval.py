"""Opt-in regression gate for the golden retrieval eval.

Skipped unless FLOWMAP_GOLDEN=1 is set and Ollama answers, because it indexes a
real corpus with a real embedding model and shells out to the CLI per query.

    FLOWMAP_GOLDEN=1 uv run pytest -m golden -q

Every (corpus, golden, baseline) triple under evals/ that exists on disk is
checked: the tracked smoke set always, the private zedbe set when its
gitignored files are present.
"""

import json
import os
from pathlib import Path

import pytest
import requests

from evals.run_eval import (
    ROOT,
    build_report,
    diff_baseline,
    ensure_index,
    load_corpus,
    manifest_hash,
    materialize_corpus,
    subprocess_cli,
    write_config,
)

pytestmark = pytest.mark.golden

EVALS = ROOT / "evals"
TOLERANCE = 0.02

_CANDIDATES = [
    ("corpus.smoke.yaml", "golden.smoke.yaml", "baseline.smoke.json"),
    ("corpus.local.yaml", "golden.local.yaml", "baseline.local.json"),
    ("corpus.yaml", "golden.yaml", "baseline.json"),
]
CASES = [
    (EVALS / c, EVALS / g, EVALS / b)
    for c, g, b in _CANDIDATES
    if (EVALS / c).exists() and (EVALS / g).exists() and (EVALS / b).exists()
]


def _ollama_up(url: str) -> bool:
    try:
        return requests.get(url.rstrip("/") + "/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.mark.skipif(not os.environ.get("FLOWMAP_GOLDEN"), reason="set FLOWMAP_GOLDEN=1 to run the golden eval")
@pytest.mark.parametrize("corpus_path,golden_path,baseline_path", CASES, ids=[c[0].stem for c in CASES])
def test_no_regression_against_baseline(corpus_path: Path, golden_path: Path, baseline_path: Path):
    corpus = load_corpus(corpus_path)
    ollama_url = corpus.embedding.get("ollama_url", "http://localhost:11434")
    if corpus.embedding.get("backend", "ollama") == "ollama" and not _ollama_up(ollama_url):
        pytest.skip(f"Ollama not reachable at {ollama_url}")

    work = EVALS / ".work" / corpus.name
    paths = materialize_corpus(corpus, work)
    config_path = write_config(corpus, work, paths)
    ensure_index(subprocess_cli, config_path, work, manifest_hash(corpus))

    report = build_report(subprocess_cli, config_path, corpus, golden_path)
    baseline = json.loads(baseline_path.read_text())

    assert baseline["corpus"]["hash"] == report["corpus"]["hash"], (
        f"baseline was cut on corpus hash {baseline['corpus']['hash']}, current is "
        f"{report['corpus']['hash']}; re-cut with --update-baseline"
    )
    assert not report["not_extracted"], f"gold symbols missing from the index: {report['not_extracted']}"

    regressions = diff_baseline(report["metrics"], baseline["metrics"], TOLERANCE)
    assert not regressions, "regressions vs baseline (tolerance %.2f):\n  " % TOLERANCE + "\n  ".join(regressions)
