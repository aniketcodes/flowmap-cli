"""File path resolution — map user-provided paths to repo-relative paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flowmap.config import RepoConfig


@dataclass
class ResolvedFile:
    repo_cfg: RepoConfig
    repo_root: Path
    abs_file: Path
    rel_file: str       # POSIX-normalized, matches git ls-files output


def resolve_file(
    file_path: str,
    repos: list[RepoConfig],
    explicit_repo: str | None = None,
) -> ResolvedFile:
    """Resolve a file path argument to a specific repo and absolute path.

    Strategy:
    1. If explicit_repo given, treat file_path as relative to that repo
    2. Auto-detect: check if absolute path is inside a configured repo
    3. Try repo/relative/path format (split on first /)

    Raises ValueError with a user-friendly message on failure.
    """
    resolved = Path(file_path).expanduser().resolve()

    if explicit_repo:
        matching = [r for r in repos if r.name == explicit_repo]
        if not matching:
            raise ValueError(f"Repo '{explicit_repo}' not found in config.")
        repo_cfg = matching[0]
        repo_root = repo_cfg.resolved_path()
        abs_file = (repo_root / file_path).resolve()
        # Prevent path traversal outside repo root
        try:
            abs_file.relative_to(repo_root)
        except ValueError:
            raise ValueError(f"Path '{file_path}' resolves outside repo root: {repo_root}")
        rel_file = str(abs_file.relative_to(repo_root)).replace("\\", "/")
        return ResolvedFile(repo_cfg=repo_cfg, repo_root=repo_root, abs_file=abs_file, rel_file=rel_file)

    # Auto-detect: find which configured repo contains this file
    for r in repos:
        rp = r.resolved_path()
        try:
            resolved.relative_to(rp)
            rel_file = str(resolved.relative_to(rp)).replace("\\", "/")
            return ResolvedFile(repo_cfg=r, repo_root=rp, abs_file=resolved, rel_file=rel_file)
        except ValueError:
            continue

    # Try treating file_path as repo/relative/path
    parts = file_path.split("/", 1)
    if len(parts) == 2:
        for r in repos:
            if r.name == parts[0]:
                repo_root = r.resolved_path()
                abs_file = (repo_root / parts[1]).resolve()
                try:
                    abs_file.relative_to(repo_root)
                except ValueError:
                    raise ValueError(f"Path '{file_path}' resolves outside repo root: {repo_root}")
                rel_file = str(abs_file.relative_to(repo_root)).replace("\\", "/")
                return ResolvedFile(repo_cfg=r, repo_root=repo_root, abs_file=abs_file, rel_file=rel_file)

    raise ValueError(f"Cannot determine repo for '{file_path}'. Use --repo flag.")
