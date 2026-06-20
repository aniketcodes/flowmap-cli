"""Indexing service — orchestrate full and incremental repo indexing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from flowmap.config import RepoConfig
from flowmap.indexer import index_files, index_repo
from flowmap.reindex import compute_incremental, get_git_status, should_full_reindex
from flowmap.state import StateDB
from flowmap.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    repo_name: str
    mode: str       # "full" | "incremental" | "skipped" | "error"
    chunks: int
    message: str


def index_changed_content(results: list[IndexResult]) -> bool:
    """Did this index run actually touch the `text` corpus?

    True only for "full"/"incremental" runs (content added/modified/deleted).
    "skipped" (up to date / no-op incremental) and "error" leave the corpus
    untouched. Used to gate the O(corpus) FTS rebuild so a no-op `index` run
    doesn't pay for a full rebuild. Note: LanceDB FTS has no partial update, so
    any real change still triggers a full rebuild — that part is inherent.
    """
    return any(r.mode in ("full", "incremental") for r in results)


def embed_chunks(
    chunks: list[dict],
    backend,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress: bool = False,
) -> list[list[float]]:
    """Batch-embed chunk texts.

    on_progress(batch_idx, total_batches) called per batch.
    show_progress=True uses tqdm for a terminal progress bar.
    """
    texts = [c["text"] for c in chunks]
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    batch_iter = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(batch_iter, desc="  Embedding", unit="batch", total=total_batches)
        except ImportError:
            pass  # tqdm not installed, continue without bar

    for i in batch_iter:
        batch = texts[i:i + batch_size]
        embs = backend.embed_documents(batch)
        all_embeddings.extend(embs)
        if on_progress:
            on_progress(i // batch_size, total_batches)
    return all_embeddings


def _full_index_repo(
    repo_name: str,
    repo_path: str,
    store: VectorStore,
    state: StateDB,
    backend,
    git_sha: str,
    git_branch: str,
    on_progress: Callable[[int, int], None] | None = None,
    on_msg: Callable[[str], None] | None = None,
    profile: str = "default",
) -> IndexResult:
    """Run a full re-index for a single repo."""
    # Step 1: Chunk and embed FIRST (failure-prone step — no store mutations yet)
    chunks = index_repo(repo_path, repo_name=repo_name)
    if not chunks:
        return IndexResult(repo_name, "full", 0, "no supported files found")

    if on_msg:
        on_msg(f"  {len(chunks)} chunks, embedding...")
    all_embeddings = embed_chunks(chunks, backend, on_progress=on_progress, show_progress=True)

    # Step 2: Upsert new data FIRST, then clean up stale chunks.
    # This closes the data-loss window: if we crash after upsert but before
    # cleanup, we have stale+new data (no loss). Pending marker handles recovery.
    state.set_meta(f"pending:{repo_name}", git_sha, profile)
    store.upsert_chunks(chunks, all_embeddings, profile=profile)

    # Step 3: Delete chunks for files no longer in the repo
    new_files = {c["file"] for c in chunks}
    stale_cleanup_ok = store.delete_stale_files(repo_name, new_files, profile=profile)

    state.set_meta("embedding_model", backend.model_name(), profile)
    state.set_meta("embedding_dims", str(backend.dims()), profile)
    state.set_repo_index(repo_name, sha=git_sha, branch=git_branch, chunks=len(chunks), profile=profile)
    # Mirror to the global repos table ONLY for the legacy default profile (back-compat).
    # Non-default profiles are fully tracked in per-profile meta, so writing the shared
    # row would just reintroduce last-writer-wins poisoning across profiles.
    if profile == "default":
        state.update_repo_indexed(
            name=repo_name,
            sha=git_sha,
            branch=git_branch,
            chunk_count=len(chunks),
        )
    # Clear the pending marker only if stale cleanup succeeded. If it failed, the
    # upsert still landed (no data loss), but orphaned chunks may remain — leaving
    # the marker set forces a clean full reindex on the next run instead of
    # silently clearing it and letting the orphans persist unnoticed.
    if stale_cleanup_ok:
        state.set_meta(f"pending:{repo_name}", "", profile)
    elif on_msg:
        on_msg(f"  Warning: stale-chunk cleanup failed for {repo_name}; "
               f"will force a full reindex next run.")

    return IndexResult(repo_name, "full", len(chunks), f"{len(chunks)} chunks indexed")


def run_index(
    store: VectorStore,
    state: StateDB,
    backend,
    targets: list[RepoConfig],
    full: bool = False,
    on_message: Callable[[str], None] | None = None,
    on_embed_progress: Callable[[int, int], None] | None = None,
    profile: str = "default",
) -> list[IndexResult]:
    """Index one or more repos into `profile`'s table. Returns per-repo results.

    Does NOT rebuild FTS/vector indexes — caller should do that once after.
    on_message(msg) is called for status updates (progress, warnings).
    on_embed_progress(batch_idx, total) is called per embedding batch.

    Staleness is tracked per profile, so indexing one embedding model never
    marks another model's index as fresh.
    """
    msg = on_message or (lambda m: None)
    results: list[IndexResult] = []

    for t in targets:
        resolved = t.resolved_path()
        if not resolved.is_dir():
            msg(f"Warning: {t.name} path does not exist: {resolved}")
            results.append(IndexResult(t.name, "error", 0, f"path not found: {resolved}"))
            continue

        state.upsert_repo(t.name, str(resolved))

        # Get current git status. Staleness is read from this profile's own
        # records; for the legacy `default` profile, fall back to the global
        # repos table so pre-profile indexes aren't needlessly re-indexed.
        git_status = get_git_status(str(resolved))
        idx = state.get_repo_index(t.name, profile)
        stored_sha = idx["sha"]
        stored_branch = idx["branch"]

        # Check for interrupted previous index
        pending = state.get_meta(f"pending:{t.name}", profile)
        force_full = bool(pending)
        if pending:
            msg(f"  {t.name}: interrupted previous index — forcing full re-index")

        # Decide: full or incremental
        needs_full = full or force_full
        if not needs_full and git_status and stored_sha:
            needs_full, reason = should_full_reindex(stored_sha, stored_branch, git_status)
            if reason == "already up to date":
                msg(f"  {t.name}: up to date ({git_status.branch}, {git_status.sha[:7]})")
                results.append(IndexResult(t.name, "skipped", 0, "up to date"))
                continue
            if needs_full and reason:
                msg(f"  {t.name}: {reason} — full re-index")
        elif not stored_sha:
            needs_full = True

        # Guard: no git status means we can't do incremental
        if not git_status and not needs_full:
            needs_full = True
            msg(f"  {t.name}: git status unavailable — full re-index")

        git_sha = git_status.sha if git_status else "unknown"
        git_branch = git_status.branch if git_status else ""

        if needs_full:
            msg(f"Indexing {t.name} ({resolved}) [full, {git_branch or 'unknown'}]...")
            result = _full_index_repo(
                t.name, str(resolved), store, state, backend,
                git_sha, git_branch, on_embed_progress, on_msg=msg, profile=profile,
            )
            msg(f"  Done: {t.name} ({result.message})")
            results.append(result)
        else:
            # Incremental re-index
            msg(f"Indexing {t.name} [incremental, {git_branch}]...")

            inc_result = compute_incremental(
                repo_path=str(resolved),
                repo_name=t.name,
                stored_sha=stored_sha,
                current=git_status,
                indexer_fn=index_files,
                store=store,
                embedding_backend=backend,
                state_db=state,
                on_progress=lambda m: msg(f"  {m}"),
                profile=profile,
            )

            if inc_result.mode == "full":
                # Fallback: incremental failed, do full
                msg(f"  {inc_result.reason} — running full re-index...")
                result = _full_index_repo(
                    t.name, str(resolved), store, state, backend,
                    git_sha, git_branch, on_embed_progress, on_msg=msg, profile=profile,
                )
                msg(f"  Done: {t.name} ({result.chunks} chunks, full re-index)")
                results.append(result)
            elif inc_result.mode == "skipped":
                msg(f"  {t.name}: {inc_result.reason}")
                results.append(IndexResult(t.name, "skipped", 0, inc_result.reason))
            else:
                msg(f"  Done: {t.name} (+{inc_result.added} -{inc_result.deleted} ~{inc_result.modified} ={inc_result.total_chunks} chunks)")
                results.append(IndexResult(t.name, "incremental", inc_result.total_chunks,
                                           f"+{inc_result.added} -{inc_result.deleted} ~{inc_result.modified}"))

    return results
