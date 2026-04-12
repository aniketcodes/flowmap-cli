"""Smoke tests for CLI commands using Click's CliRunner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from flowmap.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_dir(tmp_path):
    """Create a minimal config pointing at a real directory."""
    repo_dir = tmp_path / "my-repo"
    repo_dir.mkdir()
    (repo_dir / "hello.py").write_text("def hello():\n    pass\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "repos": [{"name": "my-repo", "path": str(repo_dir)}],
        "data_dir": str(tmp_path / "data"),
        "embedding": {"backend": "ollama", "model": "test-model"},
    }))
    return config_path


class TestInit:
    def test_init_creates_config(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        result = runner.invoke(main, ["--config", str(config_path), "init"])
        assert result.exit_code == 0
        assert config_path.exists()

    def test_init_refuses_overwrite(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing")
        result = runner.invoke(main, ["--config", str(config_path), "init"])
        assert "already exists" in result.output

    def test_init_force(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true")
        result = runner.invoke(main, ["--config", str(config_path), "init", "--force"])
        assert result.exit_code == 0


class TestRepos:
    def test_repos_add(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        result = runner.invoke(main, ["--config", str(config_path), "repos", "add", str(repo_dir)])
        assert result.exit_code == 0
        assert "my-repo" in result.output

    def test_repos_list_empty(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"repos": []}))
        result = runner.invoke(main, ["--config", str(config_path), "repos", "list"])
        assert "No repos configured" in result.output

    def test_repos_list_with_repos(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "repos", "list"])
        assert result.exit_code == 0
        assert "my-repo" in result.output

    def test_repos_paths(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "repos", "paths"])
        assert result.exit_code == 0
        assert "my-repo" in result.output


class TestStatus:
    def test_status_no_repos(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"repos": [], "data_dir": str(tmp_path / "data")}))
        result = runner.invoke(main, ["--config", str(config_path), "status"])
        assert "No repos configured" in result.output

    def test_status_not_indexed(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "status"])
        assert result.exit_code == 0
        assert "not indexed" in result.output


class TestMap:
    def test_map_empty(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "map"])
        assert result.exit_code == 0
        assert "No indexed data" in result.output

    def test_map_json_empty(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "map", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["repos"] == []


class TestSymbols:
    def test_symbols_empty(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "symbols"])
        assert result.exit_code == 0
        assert "No symbols found" in result.output


class TestCat:
    def test_cat_basic(self, runner, config_dir, tmp_path):
        result = runner.invoke(main, ["--config", str(config_dir), "cat", "hello.py", "--repo", "my-repo"])
        assert result.exit_code == 0
        assert "def hello" in result.output

    def test_cat_lines(self, runner, config_dir, tmp_path):
        result = runner.invoke(main, ["--config", str(config_dir), "cat", "hello.py", "--repo", "my-repo", "--lines", "1-1"])
        assert result.exit_code == 0
        assert "def hello" in result.output

    def test_cat_json(self, runner, config_dir, tmp_path):
        result = runner.invoke(main, ["--config", str(config_dir), "cat", "hello.py", "--repo", "my-repo", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["repo"] == "my-repo"
        assert "def hello" in data["content"]

    def test_cat_invalid_lines(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "cat", "hello.py", "--repo", "my-repo", "--lines", "abc"])
        assert result.exit_code != 0
        assert "Invalid line range" in result.output


class TestReset:
    def test_reset_requires_flag(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "reset"], input="y\n")
        assert "Specify --repo, --all, or --benchmarks" in result.output

    def test_reset_all(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "reset", "--all"], input="y\n")
        assert result.exit_code == 0


class TestVerbose:
    def test_verbose_flag_accepted(self, runner, config_dir):
        result = runner.invoke(main, ["--config", str(config_dir), "--verbose", "status"])
        assert result.exit_code == 0


class TestSearchCommand:
    def test_search_keyword_mode(self, runner, config_dir, tmp_path):
        """Keyword mode works without embedding backend — uses ripgrep only."""
        repo_dir = tmp_path / "my-repo"
        # Write a file with searchable content
        (repo_dir / "hello.py").write_text("def hello_world():\n    print('hello world')\n")
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "hello", "--mode", "keyword",
        ])
        # Should not crash — may find results or not depending on rg availability
        assert result.exit_code == 0

    def test_search_symbol_no_index(self, runner, config_dir):
        """Symbol search on empty index returns no results gracefully."""
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "myFunc", "--mode", "symbol",
        ])
        assert result.exit_code == 0
        assert "No symbols found" in result.output

    def test_search_symbol_json_empty(self, runner, config_dir):
        """Symbol search with --format json emits valid JSON on empty results."""
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "myFunc", "--mode", "symbol", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query"] == "myFunc"
        assert data["mode"] == "symbol"
        assert data["results"] == []

    def test_search_keyword_json_empty(self, runner, config_dir):
        """Keyword search with --format json emits valid JSON on empty results."""
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "xyznonexistent999", "--mode", "keyword", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query"] == "xyznonexistent999"
        assert data["mode"] == "keyword"
        assert data["results"] == []

    def test_search_keyword_regex_flag(self, runner, config_dir, tmp_path):
        """--regex flag is accepted and propagates to ripgrep (no --fixed-strings)."""
        repo_dir = tmp_path / "my-repo"
        (repo_dir / "test.py").write_text("def hello_world():\n    print('hello world')\n")
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "hello.*world", "--mode", "keyword", "--regex",
        ])
        assert result.exit_code == 0

    def test_search_hybrid_json_empty(self, runner, config_dir):
        """Hybrid search with --format json emits valid JSON even when backend unavailable."""
        result = runner.invoke(main, [
            "--config", str(config_dir), "search", "xyznonexistent999", "--format", "json",
        ])
        assert result.exit_code == 0
        # When embedding unavailable, hybrid falls back to keyword mode — still valid JSON
        assert "{" in result.output, f"No JSON found in output: {result.output!r}"
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert data["results"] == []


class TestIndexCommand:
    def test_index_dry_run_fast(self, runner, config_dir):
        """--dry-run shows file count without parsing (should be near-instant)."""
        import time
        start = time.time()
        result = runner.invoke(main, ["--config", str(config_dir), "index", "--dry-run"])
        elapsed = time.time() - start
        assert result.exit_code == 0
        assert elapsed < 5.0  # should be <1s, generous margin for CI
        assert "supported files" in result.output or "up to date" in result.output or "not a git repo" in result.output

    def test_index_repo_not_found(self, runner, config_dir):
        result = runner.invoke(main, [
            "--config", str(config_dir), "index", "--repo", "nonexistent-repo",
        ])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_index_no_repos(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("repos: []\ndata_dir: " + str(tmp_path / "data"))
        result = runner.invoke(main, ["--config", str(config_path), "index"])
        assert result.exit_code != 0
        assert "No repos configured" in result.output


class TestHistoryCommand:
    def test_history_empty_index(self, runner, config_dir):
        result = runner.invoke(main, [
            "--config", str(config_dir), "history", "someQuery",
        ])
        assert result.exit_code == 0
        assert "No history found" in result.output

    def test_history_json_empty(self, runner, config_dir):
        result = runner.invoke(main, [
            "--config", str(config_dir), "history", "someQuery", "--format", "json",
        ])
        assert result.exit_code == 0
        # Extract JSON from output (stderr warnings may be mixed in by CliRunner)
        assert "{" in result.output, f"No JSON found in output: {result.output!r}"
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert data["query"] == "someQuery"
        assert data["entries"] == []
        assert "scoped_files" in data
        assert "scoped_symbols" in data


class TestMalformedConfig:
    def test_malformed_yaml_shows_error(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("repos:\n  - name: r1\n    path: [invalid\n")
        result = runner.invoke(main, ["--config", str(config_path), "status"])
        assert result.exit_code != 0
        assert "Invalid YAML" in result.output


class TestDoctor:
    def test_doctor_no_repos(self, runner, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"repos": [], "data_dir": str(tmp_path / "data")}))
        result = runner.invoke(main, ["--config", str(config_path), "doctor"])
        assert "No repos configured" in result.output

    def test_doctor_with_repos(self, runner, config_dir):
        """Doctor runs without crashing on a valid config (embedding may fail)."""
        result = runner.invoke(main, ["--config", str(config_dir), "doctor"])
        # Should not crash — embedding backend may fail but doctor handles it
        assert "Repos:" in result.output
        assert "Embedding backend:" in result.output
