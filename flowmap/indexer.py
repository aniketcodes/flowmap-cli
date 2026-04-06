"""FlowMap indexer — walks repos, chunks files, prepares data for embedding + storage."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from flowmap.config import (
    MAX_FILE_SIZE,
    SKIP_FILENAMES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILENAMES,
    load_ignore_patterns,
)
from flowmap.parsing import Chunk, chunk_file
from flowmap.store import make_chunk_id

log = logging.getLogger(__name__)


def _git_tracked_files(repo_path: str) -> list[str] | None:
    """Get list of git-tracked files via `git ls-files`. Returns None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return [f for f in result.stdout.splitlines() if f.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _process_file(repo_path: str, repo_name: str, rel_path: str) -> list[dict]:
    """Chunk a single file and return chunk dicts. Shared by index_repo and index_files."""
    filename = os.path.basename(rel_path)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS and filename not in SUPPORTED_FILENAMES:
        return []

    filepath = os.path.join(repo_path, rel_path)
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return []
    if size > MAX_FILE_SIZE or size == 0:
        if size > MAX_FILE_SIZE:
            log.info("Skipping large file (%d bytes): %s", size, rel_path)
        return []

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except (OSError, IOError):
        return []

    if not content.strip():
        return []

    chunk_ext = ext if ext else ".txt"
    file_chunks = chunk_file(filepath, content, chunk_ext)

    chunks = []
    for i, chunk in enumerate(file_chunks):
        chunk_id = make_chunk_id(
            repo=repo_name, file=rel_path,
            symbol_name=chunk.symbol_name, chunk_type=chunk.chunk_type,
            chunk_index=i,
        )
        chunks.append({
            "id": chunk_id, "repo": repo_name, "file": rel_path,
            "file_name": filename, "extension": ext,
            "language": chunk.language, "chunk_type": chunk.chunk_type,
            "symbol_name": chunk.symbol_name, "signature": chunk.signature,
            "parent_symbol": chunk.parent_symbol, "parent_signature": chunk.parent_signature,
            "start_line": chunk.start_line, "end_line": chunk.end_line,
            "chunk_index": i, "text": chunk.text,
        })
    return chunks


def index_repo(repo_path: str, repo_name: str | None = None) -> list[dict]:
    """Walk a repo, chunk files with tree-sitter, return chunk dicts ready for embedding.

    Uses `git ls-files` to respect .gitignore. Falls back to os.walk for non-git repos.
    Additionally respects .flowmapignore patterns.
    """
    repo_path = os.path.abspath(repo_path)
    repo_name = repo_name or os.path.basename(repo_path)

    # Load .flowmapignore patterns (on top of git's own ignore)
    flowmap_ignore = load_ignore_patterns(repo_path)
    ignore_spec = None
    if flowmap_ignore:
        import pathspec
        ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", flowmap_ignore)

    # Get file list — prefer git ls-files (respects all gitignore rules perfectly)
    git_files = _git_tracked_files(repo_path)

    if git_files is not None:
        file_list = git_files
        log.info("Using git ls-files: %d tracked files", len(file_list))
    else:
        # Fallback for non-git repos: walk filesystem
        log.info("Not a git repo, falling back to filesystem walk")
        file_list = _walk_files(repo_path)

    chunks: list[dict] = []

    for rel_path in file_list:
        filename = os.path.basename(rel_path)

        if filename in SKIP_FILENAMES:
            continue
        if ignore_spec and ignore_spec.match_file(rel_path):
            continue

        chunks.extend(_process_file(repo_path, repo_name, rel_path))

    return chunks


def index_files(repo_path: str, repo_name: str, file_list: list[str]) -> list[dict]:
    """Chunk specific files (for incremental reindex). Uses shared _process_file.

    Respects .flowmapignore patterns — files matching ignore rules are skipped.
    """
    repo_path = os.path.abspath(repo_path)

    # Load .flowmapignore (same as index_repo)
    flowmap_ignore = load_ignore_patterns(repo_path)
    ignore_spec = None
    if flowmap_ignore:
        import pathspec
        ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", flowmap_ignore)

    chunks: list[dict] = []
    for rel_path in file_list:
        if ignore_spec and ignore_spec.match_file(rel_path):
            continue
        chunks.extend(_process_file(repo_path, repo_name, rel_path))
    return chunks


def _walk_files(repo_path: str) -> list[str]:
    """Fallback file walker for non-git repos. Skips common non-code directories."""
    from flowmap.config import SKIP_DIRS

    files: list[str] = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append(os.path.relpath(filepath, repo_path))
    return files


