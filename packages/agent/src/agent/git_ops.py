"""GitOperations — Git operations with flowmap-agent identity (SSH only, no gh CLI)."""

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# flowmap-agent identity
_AUTHOR_NAME = "flowmap-agent"
_AUTHOR_EMAIL = "flowmap@bot.local"


def _flowmap_env() -> dict[str, str]:
    """Return env vars with flowmap-agent identity."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = _AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = _AUTHOR_EMAIL
    env["GIT_COMMITTER_NAME"] = _AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = _AUTHOR_EMAIL
    return env


class GitOperations:
    """Git operations with flowmap-agent identity. Uses SSH, no gh CLI."""

    def generate_branch_name(self, diagnosis: str) -> str:
        """Generate deterministic branch name from diagnosis."""
        slug = re.sub(r"[^a-z0-9]+", "-", diagnosis.lower())[:40].strip("-")
        short_hash = hashlib.sha256(diagnosis.encode()).hexdigest()[:6]
        return f"flowmap/bugfix/{slug}-{short_hash}"

    def create_branch(self, repo_path: Path, branch_name: str) -> bool:
        """Create and checkout a new branch."""
        try:
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                env=_flowmap_env(),
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("create_branch_failed branch=%s stderr=%s", branch_name, result.stderr)
                return False
            logger.info("branch_created branch=%s", branch_name)
            return True
        except Exception as e:
            logger.error("create_branch_error branch=%s error=%s", branch_name, e)
            return False

    def apply_fix(self, repo_path: Path, files: dict[str, str]) -> bool:
        """Write fixed file contents to disk."""
        try:
            for rel_path, content in files.items():
                full_path = Path(repo_path) / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                logger.info("file_written path=%s", rel_path)
            return True
        except Exception as e:
            logger.error("apply_fix_error error=%s", e)
            return False

    def commit(self, repo_path: Path, message: str) -> bool:
        """Stage all changes and commit with flowmap-agent identity."""
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(repo_path),
                capture_output=True,
                timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                env=_flowmap_env(),
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("commit_failed stderr=%s", result.stderr)
                return False
            logger.info("committed message=%s", message[:50])
            return True
        except Exception as e:
            logger.error("commit_error error=%s", e)
            return False

    def push(self, repo_path: Path, branch_name: str) -> bool:
        """Push branch to origin via SSH."""
        try:
            result = subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("push_failed branch=%s stderr=%s", branch_name, result.stderr)
                return False
            logger.info("pushed branch=%s", branch_name)
            return True
        except Exception as e:
            logger.error("push_error branch=%s error=%s", branch_name, e)
            return False

    def get_pr_url(self, repo_path: Path, branch_name: str) -> str:
        """Generate GitHub PR creation URL from branch name.

        Returns a URL the user can click to create the PR manually.
        """
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ""

            url = result.stdout.strip()
            # Parse git@github.com:org/repo.git -> https://github.com/org/repo
            match = re.match(r"git@github\.com:(.+?)\.git", url)
            if match:
                return f"https://github.com/{match.group(1)}/compare/{branch_name}?expand=1"

            # Parse https://github.com/org/repo.git
            match = re.match(r"https://github\.com/(.+?)\.git", url)
            if match:
                return f"https://github.com/{match.group(1)}/compare/{branch_name}?expand=1"

            return ""
        except Exception as e:
            logger.error("get_pr_url_error error=%s", e)
            return ""
