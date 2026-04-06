"""Incremental reindexing — re-embed only changed files via git diff."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class GitStatus:
    sha: str
    branch: str


@dataclass
class FileChange:
    status: str     # A (added), M (modified), D (deleted), R (renamed)
    path: str
    old_path: str | None = None  # only for renames


@dataclass
class IncrementalResult:
    mode: str           # "incremental" | "full" | "skipped"
    reason: str
    added: int = 0
    modified: int = 0
    deleted: int = 0
    renamed: int = 0
    total_chunks: int = 0


def get_git_status(repo_path: str) -> GitStatus | None:
    """Get current SHA and branch for a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0 or branch.returncode != 0:
            return None
        return GitStatus(
            sha=sha.stdout.strip(),
            branch=branch.stdout.strip(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_changed_files(repo_path: str, old_sha: str, new_sha: str) -> list[FileChange] | None:
    """Get list of changed files between two commits via git diff --name-status.

    Returns None if old_sha is invalid (force push/rebase).
    """
    # Validate old SHA exists
    try:
        check = subprocess.run(
            ["git", "rev-parse", "--verify", old_sha],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        if check.returncode != 0:
            log.info("Old SHA %s not found (force push?), falling back to full index", old_sha[:7])
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Get diff
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", f"{old_sha}..{new_sha}"],
            cwd=repo_path, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.warning("git diff failed: %s", result.stderr[:200])
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    changes: list[FileChange] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        status = parts[0]

        if status.startswith("R"):
            # Rename: R100\told_path\tnew_path
            if len(parts) >= 3:
                changes.append(FileChange(status="R", path=parts[2], old_path=parts[1]))
        elif status in ("A", "M", "D"):
            changes.append(FileChange(status=status, path=parts[1]))
        elif status == "C":
            # Copy: treat as add
            if len(parts) >= 3:
                changes.append(FileChange(status="A", path=parts[2]))

    return changes


def should_full_reindex(
    stored_sha: str | None,
    stored_branch: str | None,
    current: GitStatus,
) -> tuple[bool, str]:
    """Determine if a full re-index is needed instead of incremental.

    Returns (should_full, reason).
    """
    if not stored_sha:
        return True, "first index"

    if stored_branch and stored_branch != current.branch:
        return True, f"branch changed ({stored_branch} → {current.branch})"

    if stored_sha == current.sha:
        return False, "already up to date"

    return False, ""


def compute_incremental(
    repo_path: str,
    repo_name: str,
    stored_sha: str,
    current: GitStatus,
    indexer_fn,
    store,
    embedding_backend,
    state_db,
    on_progress=None,
) -> IncrementalResult:
    """Run incremental reindex: only re-embed changed files.

    indexer_fn: callable(repo_path, repo_name, file_list) -> list[dict]
        A function that chunks specific files and returns chunk dicts.
    """
    changes = get_changed_files(repo_path, stored_sha, current.sha)

    if changes is None:
        # SHA invalid, need full reindex
        return IncrementalResult(mode="full", reason="stored SHA not found (force push?)")

    if not changes:
        return IncrementalResult(mode="skipped", reason="no changes since last index")

    added = [c for c in changes if c.status == "A"]
    modified = [c for c in changes if c.status == "M"]
    deleted = [c for c in changes if c.status == "D"]
    renamed = [c for c in changes if c.status == "R"]

    if on_progress:
        on_progress(f"{len(added)} added, {len(modified)} modified, {len(deleted)} deleted, {len(renamed)} renamed")

    # Set pending marker before mutations — crash recovery
    state_db.set_meta(f"pending:{repo_name}", current.sha)

    # Step 1: Chunk and embed FIRST (failure-prone step — no store mutations yet)
    # If embedding fails here, no data has been deleted.
    files_to_index = [c.path for c in added + modified + renamed]
    chunks: list[dict] = []
    all_embeddings: list[list[float]] = []
    total_chunks = 0

    if files_to_index:
        chunks = indexer_fn(repo_path, repo_name, files_to_index)
        if chunks:
            from flowmap.services.indexing import embed_chunks
            all_embeddings = embed_chunks(chunks, embedding_backend, batch_size=32)

    # Step 2: Upsert new chunks FIRST (merge_insert handles same-ID updates).
    # This closes the data-loss window: if we crash after upsert but before
    # cleanup, we have stale+new data (no loss). Pending marker handles recovery.
    if chunks and all_embeddings:
        store.upsert_chunks(chunks, all_embeddings)
        total_chunks = len(chunks)

    # Step 3: Clean up stale data (safe — upsert already succeeded)
    # Deleted files: remove all chunks (no new data exists for these)
    for c in deleted:
        store.delete_by_file(repo_name, c.path)
    # Renamed files: remove old path (new path was upserted in step 2)
    for c in renamed:
        if c.old_path:
            store.delete_by_file(repo_name, c.old_path)
    # Modified files: remove only stale chunk IDs (symbols removed or reordered)
    # Can't use delete_by_file — it would delete the just-upserted chunks too.
    new_ids_by_file: dict[str, set[str]] = {}
    for chunk in chunks:
        new_ids_by_file.setdefault(chunk["file"], set()).add(chunk["id"])
    for c in modified:
        valid_ids = new_ids_by_file.get(c.path, set())
        store.delete_stale_chunks(repo_name, c.path, valid_ids)

    # Get actual total chunk count from the store (not just the delta)
    actual_count = total_chunks
    try:
        stats = store.get_stats(known_repos=[repo_name])
        actual_count = stats["repos"].get(repo_name, total_chunks)
    except Exception:
        pass  # fall back to delta count if stats query fails

    # Update state
    state_db.update_repo_indexed(
        name=repo_name,
        sha=current.sha,
        branch=current.branch,
        chunk_count=actual_count,
    )
    # Clear pending marker after successful state update
    state_db.set_meta(f"pending:{repo_name}", "")

    return IncrementalResult(
        mode="incremental",
        reason=f"{current.sha[:7]} (was {stored_sha[:7]})",
        added=len(added),
        modified=len(modified),
        deleted=len(deleted),
        renamed=len(renamed),
        total_chunks=actual_count,
    )
