"""Pure scoring for the golden retrieval eval.

Everything here is side-effect free apart from reading the golden YAML. The
runner (evals/run_eval.py) is responsible for producing ResultRows by calling the
CLI; this module decides what counts as a hit and turns hits into metrics and
feedback text.

Semantics (see docs/PLAN_GOLDEN_EVAL_TDD.md):
- A result matches a gold target when repo and file agree and either the symbol
  names agree, the gold symbol is null (file-level gold), or the result is a
  bare line hit (no symbol, e.g. ripgrep) whose line range overlaps the gold
  symbol's known span.
- hit@1 requires a *primary* gold at rank 1. MRR and recall use any grade.
- recall@k counts distinct gold targets found in the top k, so duplicate chunks
  of the same symbol count once.
- file_hit is diagnostic only: right file, wrong symbol.
"""

from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from flowmap.search.hybrid import classify_query

GRADES = ("primary", "acceptable")
TYPES = ("identifier", "mixed", "natural_language")
TOP_K = 10

# Tags are free-form labels aggregated separately (see aggregate()["by_tag"]).
# The `blind` tag is special: the loader rejects a blind query whose content
# words overlap the name of any primary gold symbol by more than
# BLIND_MAX_OVERLAP. This keeps prose queries from being paraphrases of the
# identifier they are supposed to find, which inflates every word-matching leg.
BLIND_TAG = "blind"
BLIND_MAX_OVERLAP = 1
_STOPWORDS = frozenset("""
a an the of to for in on by and or with from at is are was were be been being it its this that
these those when whether how what where which who do does did done can could should would will
we our us you your they their them he she his her one two three some any all each per not no
into onto out up down over under again still also just only than then there here
""".split())

# (repo, file, symbol) -> (start_line, end_line), resolved once per run from
# `flowmap symbols` so bare line hits can be attributed to a symbol.
SymbolSpans = dict[tuple[str, str, str], tuple[int, int]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _norm_file(path: str) -> str:
    return path[2:] if path.startswith("./") else path


@dataclass(frozen=True)
class GoldTarget:
    repo: str
    file: str
    symbol: str | None = None
    grade: str = "primary"

    def __post_init__(self):
        object.__setattr__(self, "file", _norm_file(self.file))

    @property
    def label(self) -> str:
        sym = self.symbol if self.symbol is not None else "*"
        return f"{self.repo}/{self.file}::{sym}"


@dataclass
class GoldQuery:
    id: str
    query: str
    type: str
    gold: list[GoldTarget]
    repo: str | None = None
    rerank: bool = False
    notes: str = ""
    confirmed_by: str = ""
    tags: list[str] = field(default_factory=list)


def name_words(symbol_or_path: str) -> set[str]:
    """Split camelCase / snake_case / dotted / kebab names into lowercase words."""
    return {t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", symbol_or_path or "")}


def content_words(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOPWORDS}


def name_overlap(query: str, target: GoldTarget) -> int:
    """How many content words of the query also appear in the target's symbol
    name (or file stem for file-level gold)."""
    name = target.symbol if target.symbol is not None else Path(target.file).stem
    return len(content_words(query) & name_words(name))


@dataclass
class ResultRow:
    """One row of `flowmap search --format json`; only the fields scoring needs."""
    repo: str
    file: str
    symbol_name: str
    start_line: int
    end_line: int
    sources: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.file = _norm_file(self.file)

    @classmethod
    def from_json(cls, d: dict) -> "ResultRow":
        return cls(
            repo=d["repo"],
            file=d["file"],
            symbol_name=d.get("symbol_name") or "",
            start_line=int(d.get("start_line") or 0),
            end_line=int(d.get("end_line") or 0),
            sources=list(d.get("sources") or []),
        )

    @property
    def label(self) -> str:
        if self.symbol_name:
            return f"{self.repo}/{self.file}::{self.symbol_name}"
        return f"{self.repo}/{self.file}:{self.start_line}-{self.end_line}"


@dataclass
class QueryScore:
    id: str
    type: str
    hit_at_1: bool
    mrr: float
    recall_at_5: float
    recall_at_10: float
    first_hit_rank: int | None
    file_hit_at_10: bool
    file_hit_rank: int | None
    sources_at_hit: list[str]
    missing: list[GoldTarget]
    found: list[GoldTarget]
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def symbol_matches(name: str, wanted: str) -> bool:
    """Exact match, or dotted-suffix match so gold `hello` hits the chunker's
    qualified `Greeter.hello`. Mirrors the store's own symbol-search semantics;
    the file constraint in match() keeps this from being ambiguous."""
    return name == wanted or name.endswith("." + wanted)


def match(row: ResultRow, target: GoldTarget, spans: SymbolSpans) -> bool:
    if row.repo != target.repo or row.file != target.file:
        return False
    if target.symbol is None:
        return True
    if row.symbol_name:
        return symbol_matches(row.symbol_name, target.symbol)
    # Bare line hit: attribute via the symbol's known span.
    span = spans.get((target.repo, target.file, target.symbol))
    if span is None:
        return False
    start, end = span
    return row.start_line <= end and row.end_line >= start


def _file_only(row: ResultRow, target: GoldTarget) -> bool:
    return row.repo == target.repo and row.file == target.file


# ---------------------------------------------------------------------------
# Per-query scoring
# ---------------------------------------------------------------------------

def score_query(q: GoldQuery, rows: list[ResultRow], spans: SymbolSpans, k: int = TOP_K) -> QueryScore:
    rows = rows[:k]
    first_hit_rank: int | None = None
    sources_at_hit: list[str] = []
    found_at_rank: dict[int, set[int]] = {}  # rank -> indices of gold targets matched
    file_hit_rank: int | None = None

    for rank, row in enumerate(rows, start=1):
        hits = {i for i, t in enumerate(q.gold) if match(row, t, spans)}
        if hits:
            found_at_rank[rank] = hits
            if first_hit_rank is None:
                first_hit_rank = rank
                sources_at_hit = list(row.sources)
        elif file_hit_rank is None and any(_file_only(row, t) for t in q.gold):
            file_hit_rank = rank

    def found_within(n: int) -> set[int]:
        out: set[int] = set()
        for rank, hits in found_at_rank.items():
            if rank <= n:
                out |= hits
        return out

    n_gold = len(q.gold)
    found10 = found_within(10)
    hit_at_1 = 1 in found_at_rank and any(q.gold[i].grade == "primary" for i in found_at_rank[1])

    return QueryScore(
        id=q.id,
        type=q.type,
        hit_at_1=hit_at_1,
        mrr=(1.0 / first_hit_rank) if first_hit_rank else 0.0,
        recall_at_5=len(found_within(5)) / n_gold if n_gold else 0.0,
        recall_at_10=len(found10) / n_gold if n_gold else 0.0,
        first_hit_rank=first_hit_rank,
        file_hit_at_10=file_hit_rank is not None,
        file_hit_rank=file_hit_rank,
        sources_at_hit=sources_at_hit,
        missing=[t for i, t in enumerate(q.gold) if i not in found10],
        found=[t for i, t in enumerate(q.gold) if i in found10],
        tags=list(q.tags),
    )


# ---------------------------------------------------------------------------
# Feedback text (the μ_f payload for a prompt optimizer)
# ---------------------------------------------------------------------------

def feedback_text(q: GoldQuery, rows: list[ResultRow], score: QueryScore) -> str:
    if not score.missing:
        return ""
    lines = [f"missing: {t.label} ({t.grade})" for t in score.missing]
    top3 = ", ".join(f"{r.label} [{','.join(r.sources)}]" for r in rows[:3]) or "(no results)"
    lines.append(f"top-3 returned: {top3}")
    if score.file_hit_rank is not None:
        r = rows[score.file_hit_rank - 1]
        via = r.symbol_name or f"lines {r.start_line}-{r.end_line}"
        lines.append(f"file_hit: yes ({r.repo}/{r.file} ranked {score.file_hit_rank} via {via})")
    else:
        lines.append("file_hit: no")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRICS = ("hit_at_1", "mrr", "recall_at_5", "recall_at_10", "file_hit_at_10")


def _mean_block(scores: list[QueryScore]) -> dict:
    n = len(scores)
    block: dict = {"n": n}
    for m in _METRICS:
        block[m] = (sum(float(getattr(s, m)) for s in scores) / n) if n else 0.0
    return block


def aggregate(scores: Iterable[QueryScore]) -> dict:
    scores = list(scores)
    by_type: dict[str, list[QueryScore]] = {}
    for s in scores:
        by_type.setdefault(s.type, []).append(s)
    by_tag: dict[str, list[QueryScore]] = {}
    for s in scores:
        for t in s.tags:
            by_tag.setdefault(t, []).append(s)
    attribution = Counter(src for s in scores if s.first_hit_rank for src in s.sources_at_hit)
    return {
        "overall": _mean_block(scores),
        "by_type": {t: _mean_block(ss) for t, ss in sorted(by_type.items())},
        "by_tag": {t: _mean_block(ss) for t, ss in sorted(by_tag.items())},
        "source_attribution": dict(sorted(attribution.items())),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_golden(path: str | Path, corpus_repos: set[str] | None = None) -> list[GoldQuery]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    entries = raw.get("queries") or []
    queries: list[GoldQuery] = []
    seen: set[str] = set()

    for e in entries:
        qid = str(e.get("id", "")).strip()
        if not qid:
            raise ValueError(f"golden entry without id: {e!r}")
        if qid in seen:
            raise ValueError(f"duplicate golden id {qid}")
        seen.add(qid)

        qtype = e.get("type")
        if qtype not in TYPES:
            raise ValueError(f"{qid}: unknown type {qtype!r}; expected one of {TYPES}")
        expected = classify_query(e["query"])
        if qtype != expected:
            raise ValueError(
                f"{qid}: type {qtype!r} disagrees with classify_query() -> {expected!r} "
                f"for query {e['query']!r}"
            )

        gold_raw = e.get("gold") or []
        if not gold_raw:
            raise ValueError(f"{qid}: needs at least one gold target")
        gold: list[GoldTarget] = []
        for g in gold_raw:
            grade = g.get("grade", "primary")
            if grade not in GRADES:
                raise ValueError(f"{qid}: invalid grade {grade!r}; expected one of {GRADES}")
            if corpus_repos is not None and g["repo"] not in corpus_repos:
                raise ValueError(f"{qid}: gold repo {g['repo']!r} is not in the corpus {sorted(corpus_repos)}")
            gold.append(GoldTarget(repo=g["repo"], file=g["file"], symbol=g.get("symbol"), grade=grade))

        tags = [str(t) for t in (e.get("tags") or [])]
        if BLIND_TAG in tags:
            for t in gold:
                if t.grade != "primary":
                    continue
                ov = name_overlap(e["query"], t)
                if ov > BLIND_MAX_OVERLAP:
                    shared = sorted(content_words(e["query"]) & name_words(t.symbol or Path(t.file).stem))
                    raise ValueError(
                        f"{qid}: tagged blind but shares {ov} content words {shared} with gold "
                        f"{t.label}; rephrase without naming the thing (max {BLIND_MAX_OVERLAP})"
                    )

        queries.append(GoldQuery(
            id=qid,
            query=e["query"],
            type=qtype,
            gold=gold,
            repo=e.get("repo"),
            rerank=bool(e.get("rerank", False)),
            notes=e.get("notes", "") or "",
            confirmed_by=e.get("confirmed_by", "") or "",
            tags=tags,
        ))
    return queries
