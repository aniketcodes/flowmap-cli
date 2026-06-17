"""Timeline assembly — scope → git history → structural diff → chronological timeline."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from flowmap.history.git_ops import CommitInfo, get_file_at_commit, get_file_history, get_parent_sha, pickaxe_search
from flowmap.history.structural_diff import SymbolChange, structural_diff
from flowmap.store import VectorStore

log = logging.getLogger(__name__)

# Limits to keep query-time git ops fast
MAX_SCOPED_FILES = 5
MAX_DIFF_COMMITS = 20
DIFF_WORKERS = 4


@dataclass
class TimelineEntry:
    commit: CommitInfo
    file: str
    repo: str
    changes: list[SymbolChange] = field(default_factory=list)
    relevance: str = "direct"   # "direct" | "related"


@dataclass
class Timeline:
    query: str
    entries: list[TimelineEntry] = field(default_factory=list)
    scoped_files: list[str] = field(default_factory=list)
    scoped_symbols: list[str] = field(default_factory=list)


def _dedupe_results_to_scope(results) -> list[tuple[str, str, str]]:
    """Deduplicate search results to unique (repo, file, symbol) tuples."""
    seen: set[tuple[str, str]] = set()
    scoped: list[tuple[str, str, str]] = []
    for r in results:
        key = (r.repo, r.file)
        if key not in seen:
            seen.add(key)
            scoped.append((r.repo, r.file, r.symbol_name))
            if len(scoped) >= MAX_SCOPED_FILES:
                break
    return scoped


def _scope_from_symbol_search(
    query: str,
    store: VectorStore,
    repo_filter: str | None = None,
    profile: str = "default",
) -> list[tuple[str, str, str]]:
    """Use symbol search to find relevant (repo, file, symbol) tuples."""
    results = store.search_symbol(query, repo_filter=repo_filter, limit=10, profile=profile)
    return _dedupe_results_to_scope(results)


def _scope_from_vector_search(
    query: str,
    store: VectorStore,
    embedding_backend,
    repo_filter: str | None = None,
    profile: str = "default",
) -> list[tuple[str, str, str]]:
    """Fallback: use vector search to find relevant (repo, file, symbol) tuples."""
    try:
        query_vector = embedding_backend.embed_query(query)
        results = store.search_vector(query_vector, limit=10, repo_filter=repo_filter, profile=profile)
    except Exception as e:
        log.debug("Vector fallback failed during history scope: %s", e)
        return []
    return _dedupe_results_to_scope(results)


def _diff_commit(
    repo_path: str,
    file_path: str,
    sha: str,
    extension: str,
) -> list[SymbolChange]:
    """Get structural diff for a single commit on a file."""
    new_content = get_file_at_commit(repo_path, file_path, sha)
    parent = get_parent_sha(repo_path, sha)
    if parent:
        old_content = get_file_at_commit(repo_path, file_path, parent)
    else:
        old_content = None  # initial commit — everything is truly new

    # If parent exists but old_content is None, the file may have been
    # renamed — we can't reliably diff without the old path. Return empty
    # rather than reporting all symbols as "added" (which is misleading).
    if parent and old_content is None and new_content is not None:
        return []

    return structural_diff(old_content, new_content, extension)


def build_timeline(
    query: str,
    repo_paths: dict[str, str],
    store: VectorStore,
    since: str = "6 months ago",
    limit: int = 20,
    repo_filter: str | None = None,
    symbol_filter: str | None = None,
    embedding_backend=None,
    profile: str = "default",
) -> Timeline:
    """Build a timeline of structural changes for a query.

    Pipeline:
    1. Scope: find relevant files via symbol/search
    2. Git history: get commits per file + pickaxe search
    3. Structural diff: parse before/after with tree-sitter
    4. Assemble: sort by date, trim to limit
    """
    timeline = Timeline(query=query)

    # --- Step 1: Scope resolution ---
    search_term = symbol_filter or query
    scoped = _scope_from_symbol_search(search_term, store, repo_filter, profile)

    if not scoped and embedding_backend is not None:
        log.debug("Symbol search found nothing for '%s', falling back to vector search", search_term)
        scoped = _scope_from_vector_search(search_term, store, embedding_backend, repo_filter, profile)

    if not scoped:
        return timeline

    timeline.scoped_files = [f"{repo}/{file}" for repo, file, _ in scoped]
    timeline.scoped_symbols = [sym for _, _, sym in scoped if sym]

    # --- Step 2: Gather commit history ---
    # Collect all commits across scoped files + pickaxe
    # Key by (sha, file) so a commit touching multiple files keeps all entries
    all_commits: dict[tuple[str, str], tuple[CommitInfo, str, str]] = {}

    for repo_name, file_path, _ in scoped:
        repo_path = repo_paths.get(repo_name)
        if not repo_path:
            continue

        history = get_file_history(repo_path, file_path, since=since, limit=limit)
        if history:
            for commit in history.commits:
                key = (commit.sha, file_path)
                if key not in all_commits:
                    all_commits[key] = (commit, repo_name, file_path)

    # Pickaxe: find commits where the query string was added/removed
    for repo_name, repo_path in repo_paths.items():
        if repo_filter and repo_name != repo_filter:
            continue
        pickaxe_commits = pickaxe_search(repo_path, search_term, since=since, limit=limit)
        for commit in pickaxe_commits:
            key = (commit.sha, "")
            if key not in all_commits:
                # We don't know which file — mark as related
                all_commits[key] = (commit, repo_name, "")

    if not all_commits:
        return timeline

    # --- Step 3: Sort by date and limit ---
    sorted_commits = sorted(
        all_commits.values(),
        key=lambda x: x[0].date,
        reverse=True,
    )[:limit * 2]  # over-fetch, trim after diff

    # --- Step 4: Structural diff (parallelized) ---
    entries: list[TimelineEntry] = []
    diff_tasks: list[tuple[CommitInfo, str, str, str, str]] = []

    for commit, repo_name, file_path in sorted_commits:
        repo_path = repo_paths.get(repo_name, "")
        if not repo_path or not file_path:
            # No file info (from pickaxe) — add as metadata-only entry
            entries.append(TimelineEntry(
                commit=commit,
                file="",
                repo=repo_name,
                relevance="related",
            ))
            continue

        ext = Path(file_path).suffix.lower()
        if ext and len(diff_tasks) < MAX_DIFF_COMMITS:
            diff_tasks.append((commit, repo_name, file_path, repo_path, ext))
        else:
            entries.append(TimelineEntry(
                commit=commit,
                file=file_path,
                repo=repo_name,
                relevance="direct",
            ))

    # Run structural diffs in parallel
    if diff_tasks:
        with ThreadPoolExecutor(max_workers=DIFF_WORKERS) as executor:
            futures = {}
            for commit, repo_name, file_path, repo_path, ext in diff_tasks:
                future = executor.submit(_diff_commit, repo_path, file_path, commit.sha, ext)
                futures[future] = (commit, repo_name, file_path)

            try:
                for future in as_completed(futures, timeout=30):
                    commit, repo_name, file_path = futures[future]
                    try:
                        changes = future.result(timeout=5)
                    except Exception as e:
                        log.debug("Structural diff failed for %s: %s", commit.sha[:7], e)
                        changes = []

                    entries.append(TimelineEntry(
                        commit=commit,
                        file=file_path,
                        repo=repo_name,
                        changes=changes,
                        relevance="direct",
                    ))
            except TimeoutError:
                log.warning("Structural diff timed out — some commits may lack AST changes")
                for future, (commit, repo_name, file_path) in futures.items():
                    if not future.done():
                        future.cancel()
                        entries.append(TimelineEntry(
                            commit=commit, file=file_path, repo=repo_name,
                            relevance="direct",
                        ))

    # --- Step 5: Sort and trim ---
    entries.sort(key=lambda e: e.commit.date, reverse=True)
    timeline.entries = entries[:limit]

    return timeline
