# Golden retrieval eval

Measures FlowMap's ranking against labeled queries, through the real CLI
(`flowmap search --format json`), on a corpus pinned to exact commits.
Design and phases: `docs/PLAN_GOLDEN_EVAL_TDD.md`.

## Run

```console
# smoke: FlowMap itself, ~12 queries, proves the harness works
uv run python -m evals.run_eval --corpus evals/corpus.smoke.yaml --golden evals/golden.smoke.yaml

# corpus v1 (private, gitignored manifests)
uv run python -m evals.run_eval --corpus evals/corpus.local.yaml --golden evals/golden.local.yaml \
    --baseline evals/baseline.local.json          # first run writes the baseline
uv run python -m evals.run_eval --corpus evals/corpus.local.yaml --golden evals/golden.local.yaml \
    --baseline evals/baseline.local.json          # later runs: exit 2 on regression > --tolerance
```

Needs Ollama running with the model named in the manifest. Useful flags:
`--rerank` (all queries reranked, using the manifest's optional `reranking:
{backend, model}` block), `--in-process` (one model load per run instead of one
per query; use it with `--rerank`, the gate keeps subprocesses), `--type
natural_language`, `--id q001 --id q002`, `--reindex`, `--update-baseline`,
`--tolerance 0.02`. `evals/ablate_sources.py` runs the per-leg ablation.

What happens: each repo is cloned at its pinned commit under `evals/.work/<corpus>/repos/`,
indexed into `evals/.work/<corpus>/data/` with its own config (never `~/.flowmap`),
re-indexed only when the manifest hash changes. Every query runs as a subprocess of
the checked-out `flowmap`. A JSON report lands in `evals/reports/`.

## Files

| file | tracked | purpose |
|---|---|---|
| `corpus.smoke.yaml`, `golden.smoke.yaml`, `baseline.smoke.json` | yes | harness smoke test over FlowMap |
| `corpus.local.yaml`, `golden.local.yaml`, `baseline.local.json` | no | corpus v1 (zedbe), private |
| `corpus.yaml`, `golden.yaml`, `baseline.json` | yes (when created) | corpus v2, public harnesses, CI gate |
| `scoring.py` | yes | pure matching / metrics / feedback text |
| `run_eval.py` | yes | driver |
| `.work/`, `reports/` | no | scratch |

## Metrics

Per query over the top 10: `hit_at_1` (rank 1 is a *primary* gold), `mrr` (first hit of
any grade), `recall_at_5`, `recall_at_10` (distinct gold targets found), `file_hit_at_10`
(right file, wrong symbol; diagnostic only). Aggregated overall and per query type,
plus `source_attribution` (which legs produced first hits).

Every miss also gets a `feedback` string: what was missing, the top 3 returned, and
whether the file was hit. That is the text a prompt optimizer such as GEPA consumes.

`not_extracted` lists gold symbols the index does not know at all. Triage those as
chunker gaps, not ranking misses.

## Adding a query

```yaml
- id: q017
  query: "where is the approval policy for shell commands enforced"
  type: natural_language          # must equal classify_query(query)
  repo: null                      # optional --repo filter
  gold:
    - {repo: svc-a, file: src/policy/approval.ts, symbol: ApprovalPolicy.check, grade: primary}
    - {repo: svc-b, file: src/exec/guard.ts, symbol: guardShell, grade: acceptable}
  notes: "why this is the answer"
  confirmed_by: your-name
```

Gold symbol is the chunker's name (`Parent.method` for methods; a bare method name
also matches by suffix). `symbol: null` means any chunk in the file.

## Tests

`tests/test_eval_scoring.py` and `tests/test_eval_runner.py` run without Ollama and are
part of the normal suite.
