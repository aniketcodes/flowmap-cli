"""Tests for config loading, writing, and repo management."""

import pytest
import yaml

from flowmap.config import (
    FlowmapConfig,
    RepoConfig,
    _parse_config_dict,
    add_repo_to_config,
    load_config,
    load_ignore_patterns,
    write_default_config,
)


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "repos": [{"name": "r1", "path": "/tmp/r1"}],
            "data_dir": str(tmp_path / "data"),
            "embedding": {"backend": "ollama", "model": "test-model"},
        }))
        cfg = load_config(config_file)
        assert len(cfg.repos) == 1
        assert cfg.repos[0].name == "r1"
        assert cfg.embedding.model == "test-model"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, FlowmapConfig)
        assert cfg.repos == []

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert cfg.repos == []

    def test_load_malformed_yaml_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("repos:\n  - name: r1\n    path: [invalid\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)


class TestParseConfigDict:
    def test_full_config(self):
        cfg = _parse_config_dict({
            "repos": [{"name": "a", "path": "/a"}, {"name": "b", "path": "/b"}],
            "data_dir": "/data",
            "embedding": {"backend": "sentence-transformers", "model": "my-model"},
            "reranking": {"enabled": True, "model": "my-reranker"},
        })
        assert len(cfg.repos) == 2
        assert cfg.embedding.backend == "sentence-transformers"
        assert cfg.reranking.enabled is True

    def test_empty_dict_uses_defaults(self):
        cfg = _parse_config_dict({})
        assert cfg.repos == []
        assert cfg.embedding.backend == "ollama"
        assert cfg.reranking.enabled is False


class TestWriteDefaultConfig:
    def test_creates_config(self, tmp_path):
        path = write_default_config(tmp_path / "config.yaml")
        assert path.exists()
        raw = yaml.safe_load(path.read_text())
        assert "embedding" in raw or "repos" in path.read_text()

    def test_refuses_overwrite_without_force(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("existing")
        with pytest.raises(FileExistsError):
            write_default_config(path)

    def test_force_overwrite_preserves_repos(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump({"repos": [{"name": "keep", "path": "/keep"}]}))
        write_default_config(path, force=True)
        raw = yaml.safe_load(path.read_text())
        assert any(r["name"] == "keep" for r in raw.get("repos", []))


class TestAddRepoToConfig:
    def test_add_repo(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        repo = add_repo_to_config(str(repo_dir), config_path=config_path)
        assert repo.name == "my-repo"

        raw = yaml.safe_load(config_path.read_text())
        assert any(r["name"] == "my-repo" for r in raw["repos"])

    def test_add_duplicate_raises(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        add_repo_to_config(str(repo_dir), config_path=config_path)
        with pytest.raises(ValueError, match="already in config"):
            add_repo_to_config(str(repo_dir), config_path=config_path)

    def test_add_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            add_repo_to_config(str(tmp_path / "nope"), config_path=tmp_path / "c.yaml")


class TestIgnorePatterns:
    def test_load_ignore_patterns(self, tmp_path):
        (tmp_path / ".flowmapignore").write_text("*.log\n# comment\n\nbuild/\n")
        patterns = load_ignore_patterns(tmp_path)
        assert patterns == ["*.log", "build/"]

    def test_no_ignore_file(self, tmp_path):
        assert load_ignore_patterns(tmp_path) == []


class TestRepoConfig:
    def test_resolved_path(self, tmp_path):
        rc = RepoConfig(name="test", path=str(tmp_path))
        assert rc.resolved_path() == tmp_path.resolve()
