"""End-to-end integration tests — real git repo, real tree-sitter, real LanceDB, mock embeddings."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from flowmap.cli import main
from tests.conftest import MockBackend


@pytest.fixture
def e2e_setup(tmp_path):
    """Create a real git repo with Python files, config, and return config path."""
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()

    # Write Python files with known symbols
    (repo_dir / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(x, y):\n    return x * y\n"
    )
    (repo_dir / "greeter.py").write_text(
        "class Greeter:\n"
        "    def hello(self, name):\n"
        "        return f'Hello {name}'\n\n"
        "    def goodbye(self, name):\n"
        "        return f'Goodbye {name}'\n"
    )

    # Init git repo
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir, capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
    )

    # Write config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "repos": [{"name": "test-repo", "path": str(repo_dir)}],
        "data_dir": str(tmp_path / "data"),
        "embedding": {"backend": "ollama", "model": "test:mock"},
    }))

    return config_path


def _run_index(runner, config_path):
    """Run flowmap index with mocked embedding backend."""
    with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
        result = runner.invoke(main, ["--config", str(config_path), "index", "--full"])
    return result


class TestEndToEnd:
    def test_index_then_search_symbol(self, e2e_setup):
        runner = CliRunner()
        idx_result = _run_index(runner, e2e_setup)
        assert idx_result.exit_code == 0, idx_result.output
        assert "Done" in idx_result.output

        # Search for a known symbol
        result = runner.invoke(main, [
            "--config", str(e2e_setup), "search", "add", "--mode", "symbol",
        ])
        assert result.exit_code == 0
        assert "add" in result.output

    def test_index_then_map(self, e2e_setup):
        runner = CliRunner()
        _run_index(runner, e2e_setup)

        result = runner.invoke(main, [
            "--config", str(e2e_setup), "map", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["repos"]) >= 1
        repo = data["repos"][0]
        assert repo["files"] >= 2
        # Should have found at least the Greeter class
        class_names = [c["name"] for c in repo["classes"]]
        assert "Greeter" in class_names

    def test_index_then_cat_symbol(self, e2e_setup):
        runner = CliRunner()
        _run_index(runner, e2e_setup)

        result = runner.invoke(main, [
            "--config", str(e2e_setup), "cat", "math_utils.py",
            "--repo", "test-repo", "--symbol", "multiply",
        ])
        assert result.exit_code == 0
        assert "multiply" in result.output
        assert "return x * y" in result.output

    def test_incremental_index(self, e2e_setup):
        """Full index → add new file → incremental index → new symbol searchable."""
        runner = CliRunner()
        # Initial full index
        idx_result = _run_index(runner, e2e_setup)
        assert idx_result.exit_code == 0

        # Add a new file and commit
        import yaml
        cfg = yaml.safe_load(e2e_setup.read_text())
        repo_dir = cfg["repos"][0]["path"]

        Path(repo_dir, "new_utils.py").write_text(
            "def brand_new_function():\n    return 'I am new'\n"
        )
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com",
        }
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, env=env)
        subprocess.run(["git", "commit", "-m", "add new file"], cwd=repo_dir, capture_output=True, env=env)

        # Incremental index (no --full)
        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
            inc_result = runner.invoke(main, ["--config", str(e2e_setup), "index"])
        assert inc_result.exit_code == 0

        # New symbol should be searchable
        result = runner.invoke(main, [
            "--config", str(e2e_setup), "search", "brand_new_function", "--mode", "symbol",
        ])
        assert result.exit_code == 0
        assert "brand_new_function" in result.output

        # Old symbols should still exist
        result2 = runner.invoke(main, [
            "--config", str(e2e_setup), "search", "add", "--mode", "symbol",
        ])
        assert result2.exit_code == 0
        assert "add" in result2.output

    def test_index_then_search_hybrid(self, e2e_setup):
        """Default hybrid search works end-to-end (semantic + keyword + symbol)."""
        runner = CliRunner()
        _run_index(runner, e2e_setup)

        with patch("flowmap.embeddings.create_backend", return_value=MockBackend()):
            result = runner.invoke(main, [
                "--config", str(e2e_setup), "search", "hello",
            ])
        assert result.exit_code == 0

    def test_full_index_preserves_data_on_embed_failure(self, e2e_setup):
        """If embedding fails during full re-index, existing data is NOT deleted."""
        runner = CliRunner()

        # First: successful full index
        idx_result = _run_index(runner, e2e_setup)
        assert idx_result.exit_code == 0

        # Verify data exists
        result = runner.invoke(main, [
            "--config", str(e2e_setup), "search", "add", "--mode", "symbol",
        ])
        assert "add" in result.output

        # Second: attempt full re-index with failing embedding backend
        failing_backend = MagicMock()
        failing_backend.model_name.return_value = "test:mock"
        failing_backend.dims.return_value = 32
        failing_backend.embed_documents.side_effect = ConnectionError("Ollama crashed")

        with patch("flowmap.embeddings.create_backend", return_value=failing_backend):
            bad_result = runner.invoke(main, [
                "--config", str(e2e_setup), "index", "--full",
            ])
        # The index command should fail (embedding error)
        assert bad_result.exit_code != 0 or "Error" in bad_result.output or "Ollama" in bad_result.output

        # Original data should still be searchable
        result2 = runner.invoke(main, [
            "--config", str(e2e_setup), "search", "add", "--mode", "symbol",
        ])
        assert result2.exit_code == 0
        assert "add" in result2.output
