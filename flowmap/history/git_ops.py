"""Git operations for temporal queries — log, show, pickaxe search."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    sha: str
    author: str
    date: str       # ISO 8601
    message: str


@dataclass
class FileHistory:
    file: str
    commits: list[CommitInfo] = field(default_factory=list)


def _parse_log_output(output: str) -> list[CommitInfo]:
    """Parse pipe-delimited git log output into CommitInfo list."""
    commits = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\x00", 3)
        if len(parts) < 4:
            continue
        commits.append(CommitInfo(
            sha=parts[0],
            author=parts[1],
            date=parts[2],
            message=parts[3],
        ))
    return commits


_LOG_FORMAT = "%H%x00%an%x00%aI%x00%s"


def get_file_history(
    repo_path: str,
    file_path: str,
    since: str | None = None,
    limit: int = 50,
) -> FileHistory | None:
    """Get commit history for a file. --follow tracks renames.

    Returns None if git command fails (not a git repo, file not tracked, etc.).
    """
    cmd = [
        "git", "log",
        f"--pretty=format:{_LOG_FORMAT}",
        "--follow",
        f"-{limit}",
    ]
    if since:
        cmd.append(f"--since={since}")
    cmd += ["--", file_path]

    try:
        result = subprocess.run(
            cmd, cwd=repo_path,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.debug("get_file_history failed: %s", result.stderr[:200])
            return None
        return FileHistory(
            file=file_path,
            commits=_parse_log_output(result.stdout),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("get_file_history error: %s", e)
        return None


def get_file_at_commit(
    repo_path: str,
    file_path: str,
    sha: str,
) -> str | None:
    """Get file content at a specific commit via git show.

    Returns None if the file didn't exist at that commit.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{sha}:{file_path}"],
            cwd=repo_path,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_parent_sha(repo_path: str, sha: str) -> str | None:
    """Get first parent SHA of a commit. Returns None for initial commits."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{sha}^"],
            cwd=repo_path,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def pickaxe_search(
    repo_path: str,
    search_string: str,
    since: str | None = None,
    limit: int = 30,
) -> list[CommitInfo]:
    """Find commits where search_string was added or removed (git log -S).

    Returns empty list on failure.
    """
    cmd = [
        "git", "log",
        "-S", search_string,
        f"--pretty=format:{_LOG_FORMAT}",
        f"-{limit}",
    ]
    if since:
        cmd.append(f"--since={since}")
    cmd.append("--")  # end of options — prevents search_string being parsed as flag

    try:
        result = subprocess.run(
            cmd, cwd=repo_path,
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log.debug("pickaxe_search failed: %s", result.stderr[:200])
            return []
        return _parse_log_output(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("pickaxe_search error: %s", e)
        return []
