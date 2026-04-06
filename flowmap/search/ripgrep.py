"""Ripgrep wrapper — live keyword search over the filesystem."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from flowmap.config import SKIP_DIRS, SKIP_FILENAMES, load_ignore_patterns

log = logging.getLogger(__name__)


@dataclass
class RgResult:
    repo: str
    file: str           # relative to repo root
    line: int
    text: str


def is_available() -> bool:
    """Check if ripgrep (rg) is installed."""
    return shutil.which("rg") is not None


def _build_exclusion_args(repo_paths: dict[str, str]) -> list[str]:
    """Derive rg exclusion flags from the same config the indexer uses.

    Unifies exclusion rules across ripgrep, vector, and symbol search channels.
    Does NOT filter by SUPPORTED_EXTENSIONS — ripgrep finding matches in files
    the indexer skipped (e.g., .env, .cfg) is useful; RRF naturally deprioritizes
    single-channel hits.
    """
    args: list[str] = []

    for d in sorted(SKIP_DIRS):
        args.extend(["--glob", f"!**/{d}/**"])

    for f in sorted(SKIP_FILENAMES):
        args.extend(["--glob", f"!**/{f}"])

    # Per-repo .flowmapignore — only apply when searching a single repo
    # to avoid repo A's ignore patterns suppressing results in repo B
    ignore_files = [
        Path(rp) / ".flowmapignore"
        for rp in repo_paths.values()
        if (Path(rp) / ".flowmapignore").exists()
    ]
    if len(ignore_files) <= 1:
        for f in ignore_files:
            args.extend(["--ignore-file", str(f)])

    return args


def rg_search(
    query: str,
    repo_paths: dict[str, str],
    limit: int = 50,
    timeout: float = 10.0,
) -> list[RgResult]:
    """Run ripgrep across all repos. Returns live filesystem results.

    repo_paths: {repo_name: repo_path}
    Returns at most `limit` match results total (not per-file).
    Gracefully returns empty list if rg is not installed or fails.
    """
    if not repo_paths:
        return []

    if not is_available():
        log.warning("ripgrep (rg) not installed. Keyword search unavailable.")
        return []

    paths = list(repo_paths.values())

    cmd = [
        "rg",
        "--json",
        "--fixed-strings",           # literal string match, not regex
        "--max-columns", "500",      # truncate very long lines
        "--no-heading",
        "--no-binary",               # skip binary files
        *_build_exclusion_args(repo_paths),
        query,
        *paths,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.warning("ripgrep timed out after %.1fs", timeout)
        return []
    except FileNotFoundError:
        log.warning("ripgrep binary not found")
        return []

    # Exit codes: 0 = matches found, 1 = no matches, 2 = error
    if result.returncode == 2:
        log.warning("ripgrep error: %s", result.stderr[:200] if result.stderr else "unknown")
        return []
    if result.returncode == 1:
        return []

    return _parse_json_output(result.stdout, repo_paths, limit)

def _parse_json_output(
    output: str,
    repo_paths: dict[str, str],
    limit: int,
) -> list[RgResult]:
    """Parse rg --json output into RgResult objects."""
    if limit <= 0:
        return []

    # Build reverse map: absolute path prefix -> repo name
    path_to_repo: list[tuple[str, str]] = []
    for repo_name, repo_path in repo_paths.items():
        prefix = repo_path.rstrip("/") + "/"
        path_to_repo.append((prefix, repo_name))
    path_to_repo.sort(key=lambda x: -len(x[0]))

    results: list[RgResult] = []

    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if entry.get("type") == "match":
            results.append(_build_result(entry["data"], path_to_repo))
            if len(results) >= limit:
                return results

    return results


def _build_result(
    match_data: dict,
    path_to_repo: list[tuple[str, str]],
) -> RgResult:
    """Convert a parsed rg match entry to RgResult."""
    abs_path = match_data["path"]["text"]
    line_text = match_data["lines"]["text"].rstrip("\n")
    line_num = match_data["line_number"]

    # Map absolute path to repo + relative path
    repo_name = ""
    rel_path = abs_path
    for prefix, name in path_to_repo:
        if abs_path.startswith(prefix):
            repo_name = name
            rel_path = abs_path[len(prefix):]
            break

    return RgResult(
        repo=repo_name,
        file=rel_path,
        line=line_num,
        text=line_text,
    )
