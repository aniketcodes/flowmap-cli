"""4-way hybrid search (ripgrep + BM25/FTS + vector + symbol) with RRF fusion + optional Qwen3 reranking."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field

import os as _os

from flowmap.search.ripgrep import RgResult, rg_search
from flowmap.store import SearchResult, VectorStore

log = logging.getLogger(__name__)

# Set once at module level — avoids per-call env var mutation (not thread-safe)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TQDM_DISABLE", "1")

RRF_K = 60  # Standard RRF constant (tunable — evaluate against golden set)



# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    repo: str
    file: str
    start_line: int
    end_line: int
    text: str
    score: float             # final score (rerank_score if reranked, else rrf_score)
    rrf_score: float
    rerank_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    symbol_name: str = ""
    chunk_type: str = ""
    signature: str = ""
    parent_symbol: str = ""
    parent_signature: str = ""
    language: str = ""
    match_type: str = ""     # "semantic" | "keyword" | "symbol" | "combined"


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

def classify_query(query: str) -> str:
    """Classify query as identifier, mixed, or natural_language."""
    tokens = query.split()

    if len(tokens) == 1:
        token = tokens[0]
        has_camel = bool(re.search(r'[a-z][A-Z]', token))
        has_snake = '_' in token
        has_dot = '.' in token
        has_pascal = bool(re.match(r'^[A-Z][a-z]', token)) and len(token) > 1
        has_allcaps = token.isupper() and len(token) > 1 and token.isalpha()
        if has_camel or has_snake or has_dot or has_pascal or has_allcaps:
            return "identifier"
        return "natural_language"

    code_tokens = sum(1 for t in tokens if (
        re.search(r'[a-z][A-Z]', t) or
        '_' in t or
        '.' in t or
        (re.match(r'^[A-Z][a-z]', t) and len(t) > 2)
    ))
    if code_tokens > 0:
        return "mixed"

    return "natural_language"


# Source weights by query type. "fts" = BM25 leg (ranked lexical, complements
# ripgrep's exact match). These are hand-set and un-tuned: validated only on a
# ~31-query local set, so small deltas (fts 0.8 vs 1.0) are noise, not optima —
# re-run a weight sweep before trusting any value. Grid shape is pinned by
# test_weights_grid_shape.
_WEIGHTS: dict[str, dict[str, float]] = {
    "identifier":       {"ripgrep": 1.5, "symbol": 2.0, "semantic": 0.5, "fts": 1.0},
    "mixed":            {"ripgrep": 1.0, "symbol": 1.0, "semantic": 1.0, "fts": 1.0},
    "natural_language":  {"ripgrep": 0.5, "symbol": 0.3, "semantic": 2.0, "fts": 0.8},
}


# ---------------------------------------------------------------------------
# Reranking (Qwen3-Reranker; see flowmap/search/qwen_ollama_reranker.py)
# ---------------------------------------------------------------------------

def was_reranked(results: list[HybridResult]) -> bool:
    """True when a reranker actually scored these results. Every backend leaves
    rerank_score at 0.0 on failure/timeout/no-answer and sorts descending on
    success, so a real rerank always shows a positive score at the top."""
    return any(r.rerank_score > 0 for r in results)


def _rerank_with_backend(backend: str, query: str, candidates: list[HybridResult], model_name: str,
                         ollama_url: str = "http://localhost:11434") -> list[HybridResult]:
    """Dispatch to the configured reranker. Unknown backends fall back to qwen_ollama."""
    if backend == "qwen_direct":
        from flowmap.search.qwen_reranker import rerank_qwen
        return rerank_qwen(query, candidates, model_name)
    if backend != "qwen_ollama":
        log.warning("Unknown reranking backend %r; using qwen_ollama", backend)
    from flowmap.search.qwen_ollama_reranker import rerank_qwen_ollama
    return rerank_qwen_ollama(query, candidates, model_name, ollama_url)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    repo_paths: dict[str, str],
    embedding_backend,
    store: VectorStore,
    limit: int = 10,
    repo_filter: str | None = None,
    reranking_enabled: bool = False,
    reranking_model: str = "hf.co/Mungert/Qwen3-Reranker-0.6B-GGUF:Q8_0",
    reranking_backend: str = "qwen_ollama",
    reranking_ollama_url: str = "http://localhost:11434",
    regex: bool = False,
    profile: str = "default",
) -> list[HybridResult]:
    """Run all four search methods in parallel, fuse with weighted RRF, optionally rerank.

    Pipeline:
    1. Parallel: ripgrep (live keyword) + BM25/FTS (ranked lexical) + semantic (vector) + symbol (exact match)
    2. Normalize: map ripgrep lines to containing chunks (dedup BEFORE scoring)
    3. Score: weighted Reciprocal Rank Fusion
    4. Rerank: Qwen3-Reranker on top-30 candidates (if enabled)
    5. Return top-N results
    """
    query_type = classify_query(query)
    weights = _WEIGHTS[query_type]

    # --- Step 1: Run searches in parallel ---
    rg_results: list[RgResult] = []
    semantic_results: list[SearchResult] = []
    symbol_results: list[SearchResult] = []
    fts_results: list[SearchResult] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        futures[executor.submit(
            rg_search, query, repo_paths, limit=50, timeout=15.0, regex=regex
        )] = "ripgrep"

        def _semantic():
            query_vector = embedding_backend.embed_query(query)
            return store.search_vector(query_vector, limit=30, repo_filter=repo_filter, profile=profile)
        futures[executor.submit(_semantic)] = "semantic"

        # BM25/FTS leg — runs for every query type (complements ripgrep + semantic)
        futures[executor.submit(
            store.search_fts, query, limit=30, repo_filter=repo_filter, profile=profile
        )] = "fts"

        if query_type in ("identifier", "mixed"):
            futures[executor.submit(
                store.search_symbol, query, repo_filter, limit=10, profile=profile
            )] = "symbol"

        try:
            for future in as_completed(futures, timeout=20):
                source = futures[future]
                try:
                    result = future.result(timeout=5)
                    if source == "ripgrep":
                        rg_results = result
                    elif source == "semantic":
                        semantic_results = result
                    elif source == "symbol":
                        symbol_results = result
                    elif source == "fts":
                        fts_results = result
                except Exception as e:
                    log.warning("Search source '%s' failed: %s", source, e)
        except TimeoutError:
            log.warning("Search timed out after 20s. Proceeding with available results.")

    # --- Step 2: Map ripgrep lines to chunks (batch: one query per unique file) ---
    from collections import defaultdict

    rg_by_file: dict[tuple[str, str], list[RgResult]] = defaultdict(list)
    for rg in rg_results:
        rg_by_file[(rg.repo, rg.file)].append(rg)

    # Fetch all chunks per unique (repo, file) — turns N serial queries into M (M = unique files)
    _file_chunks_cache: dict[tuple[str, str], list[SearchResult]] = {}
    for repo_file_key in rg_by_file:
        _file_chunks_cache[repo_file_key] = store.get_chunks_for_file(*repo_file_key, profile=profile)

    rg_chunk_cache: dict[tuple, SearchResult | None] = {}
    for (repo_name, file_path), file_rg_results in rg_by_file.items():
        file_chunks = _file_chunks_cache.get((repo_name, file_path), [])
        for rg in file_rg_results:
            matched = None
            for chunk in file_chunks:
                if chunk.start_line <= rg.line <= chunk.end_line:
                    if matched is None or (chunk.end_line - chunk.start_line) < (matched.end_line - matched.start_line):
                        matched = chunk  # prefer smallest span
            rg_chunk_cache[(rg.repo, rg.file, rg.line)] = matched

    # --- Step 3: Build unified entries (dedup before scoring) ---
    unified: dict[tuple, _UnifiedEntry] = {}

    # Deduplicate semantic results by key
    seen_semantic: set[tuple] = set()
    for sr in semantic_results:
        key = (sr.repo, sr.file, sr.start_line)
        if key not in seen_semantic:
            seen_semantic.add(key)
            entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
            entry.sources.add("semantic")

    # Deduplicate symbol results by key
    seen_symbol: set[tuple] = set()
    for sr in symbol_results:
        key = (sr.repo, sr.file, sr.start_line)
        if key not in seen_symbol:
            seen_symbol.add(key)
            entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
            entry.sources.add("symbol")

    # Deduplicate FTS/BM25 results by key
    seen_fts: set[tuple] = set()
    for sr in fts_results:
        key = (sr.repo, sr.file, sr.start_line)
        if key not in seen_fts:
            seen_fts.add(key)
            entry = unified.setdefault(key, _UnifiedEntry.from_search_result(sr))
            entry.sources.add("fts")

    for rg in rg_results:
        cache_key = (rg.repo, rg.file, rg.line)
        chunk = rg_chunk_cache[cache_key]
        if chunk:
            key = (chunk.repo, chunk.file, chunk.start_line)
            entry = unified.setdefault(key, _UnifiedEntry.from_search_result(chunk))
            entry.sources.add("ripgrep")
        else:
            key = (rg.repo, rg.file, rg.line)
            if key not in unified:
                unified[key] = _UnifiedEntry(
                    repo=rg.repo, file=rg.file,
                    start_line=rg.line, end_line=rg.line,
                    text=rg.text, symbol_name="", chunk_type="",
                    signature="", parent_symbol="", parent_signature="",
                    language="", sources={"ripgrep"},
                )

    # --- Step 4: RRF scoring ---
    # Build per-source ranked lists (deduplicated)
    source_ranks: dict[str, list[tuple]] = {
        "semantic": list(dict.fromkeys(
            (sr.repo, sr.file, sr.start_line) for sr in semantic_results
        )),
        "symbol": list(dict.fromkeys(
            (sr.repo, sr.file, sr.start_line) for sr in symbol_results
        )),
        "fts": list(dict.fromkeys(
            (sr.repo, sr.file, sr.start_line) for sr in fts_results
        )),
        "ripgrep": [],
    }

    # Ripgrep: use cached chunk mapping
    seen_rg_keys: set[tuple] = set()
    for rg in rg_results:
        cache_key = (rg.repo, rg.file, rg.line)
        chunk = rg_chunk_cache[cache_key]
        key = (chunk.repo, chunk.file, chunk.start_line) if chunk else (rg.repo, rg.file, rg.line)
        if key not in seen_rg_keys:
            seen_rg_keys.add(key)
            source_ranks["ripgrep"].append(key)

    # Compute weighted RRF scores
    rrf_scores: dict[tuple, float] = {}
    for source, ranked_keys in source_ranks.items():
        w = weights.get(source, 1.0)
        for rank, key in enumerate(ranked_keys):
            rrf_scores.setdefault(key, 0.0)
            rrf_scores[key] += w * (1.0 / (RRF_K + rank + 1))  # 1-indexed rank

    # --- Step 5: Build scored results ---
    scored: list[HybridResult] = []
    for key, entry in unified.items():
        rrf = rrf_scores.get(key, 0.0)
        sources = sorted(entry.sources)
        match_type = "combined" if len(sources) > 1 else sources[0] if sources else ""

        scored.append(HybridResult(
            repo=entry.repo,
            file=entry.file,
            start_line=entry.start_line,
            end_line=entry.end_line,
            text=entry.text,
            score=rrf,
            rrf_score=rrf,
            sources=sources,
            symbol_name=entry.symbol_name,
            chunk_type=entry.chunk_type,
            signature=entry.signature,
            parent_symbol=entry.parent_symbol,
            parent_signature=entry.parent_signature,
            language=entry.language,
            match_type=match_type,
        ))

    scored.sort(key=lambda r: r.score, reverse=True)

    # --- Step 6: Reranking (on top-30 RRF candidates) ---
    if reranking_enabled and scored:
        top_candidates = scored[:30]
        reranked = _rerank_with_backend(reranking_backend, query, top_candidates, reranking_model, reranking_ollama_url)
        # Merge: reranked top-30 + remaining unranked
        remaining = scored[30:]
        scored = reranked + remaining

    return scored[:limit]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

@dataclass
class _UnifiedEntry:
    repo: str
    file: str
    start_line: int
    end_line: int
    text: str
    symbol_name: str
    chunk_type: str
    signature: str
    parent_symbol: str
    parent_signature: str
    language: str
    sources: set = field(default_factory=set)

    @classmethod
    def from_search_result(cls, sr: SearchResult) -> _UnifiedEntry:
        return cls(
            repo=sr.repo, file=sr.file,
            start_line=sr.start_line, end_line=sr.end_line,
            text=sr.text, symbol_name=sr.symbol_name,
            chunk_type=sr.chunk_type, signature=sr.signature,
            parent_symbol=sr.parent_symbol, parent_signature=sr.parent_signature,
            language=sr.language, sources=set(),
        )
