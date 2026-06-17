"""Tests for config loading, writing, and repo management."""

import pytest
import yaml

from flowmap.config import (
    EmbeddingConfig,
    EmbeddingProfile,
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


class TestEmbeddingProfiles:
    """Profile bundles: `active` selects a (backend, model) pair; `profile_name`
    maps to the store table. Flat config stays back-compatible as the `default`
    profile so the existing `code_index` table is never orphaned."""

    def test_flat_config_is_default_profile(self):
        """Legacy flat embedding block → implicit `default` profile."""
        cfg = _parse_config_dict({
            "embedding": {"backend": "ollama", "model": "qwen3-embedding:0.6b"},
        })
        assert cfg.embedding.active == "default"
        assert cfg.embedding.profile_name == "default"
        assert cfg.embedding.model == "qwen3-embedding:0.6b"
        # Flat form is exposed as a single named profile too
        assert "default" in cfg.embedding.profiles
        assert cfg.embedding.profiles["default"].model == "qwen3-embedding:0.6b"

    def test_no_embedding_block_defaults(self):
        cfg = _parse_config_dict({})
        assert cfg.embedding.active == "default"
        assert cfg.embedding.profile_name == "default"
        assert cfg.embedding.backend == "ollama"

    def test_profiles_map_active_selects_values(self):
        """Active profile's backend/model/url are reflected on the top-level
        fields so existing call sites (cfg.embedding.model) keep working."""
        cfg = _parse_config_dict({
            "embedding": {
                "active": "qwen4b",
                "profiles": {
                    "qwen06b": {"backend": "ollama", "model": "qwen3-embedding:0.6b"},
                    "qwen4b": {"backend": "ollama", "model": "qwen3-embedding:4b"},
                },
            },
        })
        assert cfg.embedding.active == "qwen4b"
        assert cfg.embedding.profile_name == "qwen4b"
        assert cfg.embedding.model == "qwen3-embedding:4b"
        assert cfg.embedding.backend == "ollama"
        assert set(cfg.embedding.profiles) == {"qwen06b", "qwen4b"}
        assert cfg.embedding.profiles["qwen06b"].model == "qwen3-embedding:0.6b"

    def test_profiles_map_single_profile_active_optional(self):
        """One profile, no explicit `active` → that profile is active."""
        cfg = _parse_config_dict({
            "embedding": {
                "profiles": {
                    "only": {"backend": "ollama", "model": "m1"},
                },
            },
        })
        assert cfg.embedding.active == "only"
        assert cfg.embedding.model == "m1"

    def test_profiles_map_missing_active_raises(self):
        with pytest.raises(ValueError, match="active.*not.*defined|unknown.*profile"):
            _parse_config_dict({
                "embedding": {
                    "active": "ghost",
                    "profiles": {
                        "qwen06b": {"backend": "ollama", "model": "m1"},
                    },
                },
            })

    def test_profile_inherits_ollama_url_default(self):
        cfg = _parse_config_dict({
            "embedding": {
                "profiles": {"p": {"model": "m1"}},
            },
        })
        assert cfg.embedding.profiles["p"].ollama_url == "http://localhost:11434"
        assert cfg.embedding.profiles["p"].backend == "ollama"


class TestProfileNameValidation:
    @pytest.mark.parametrize("bad", ["Bad", "a b", "a.b", "-leading", "_leading", "a__b", "a/b", ""])
    def test_invalid_profile_name_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid embedding profile name"):
            _parse_config_dict({"embedding": {"active": bad, "profiles": {bad: {"model": "m"}}}})

    @pytest.mark.parametrize("good", ["qwen06b", "qwen-4b", "a", "a_b", "p1"])
    def test_valid_profile_name_accepted(self, good):
        cfg = _parse_config_dict({"embedding": {"profiles": {good: {"model": "m"}}}})
        assert good in cfg.embedding.profiles


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

    def test_force_overwrite_backs_up_and_warns_on_malformed(self, tmp_path, capsys):
        path = tmp_path / "config.yaml"
        path.write_text("embedding:\n  profiles:\n    p1: {model: [unterminated\n")  # malformed YAML
        write_default_config(path, force=True)
        err = capsys.readouterr().err
        assert "could not parse existing config" in err
        # Original content preserved in a .bak rather than silently lost.
        backup = path.with_suffix(".yaml.bak")
        assert backup.exists()
        assert "unterminated" in backup.read_text()

    def test_force_overwrite_preserves_embedding_profiles_and_data_dir(self, tmp_path):
        """init --force must not destroy a profiles config, custom data_dir, or reranking —
        even when there are no repos (the gate must not skip preservation)."""
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump({
            "data_dir": "/custom/data",
            "embedding": {
                "active": "qwen4b",
                "profiles": {
                    "qwen06b": {"backend": "ollama", "model": "qwen3-embedding:0.6b"},
                    "qwen4b": {"backend": "ollama", "model": "qwen3-embedding:4b"},
                },
            },
            "reranking": {"enabled": True, "model": "my-reranker"},
        }))
        write_default_config(path, force=True)

        cfg = load_config(path)
        assert cfg.embedding.active == "qwen4b"
        assert set(cfg.embedding.profiles) == {"qwen06b", "qwen4b"}
        assert cfg.embedding.model == "qwen3-embedding:4b"
        assert cfg.reranking.enabled is True
        assert cfg.reranking.model == "my-reranker"
        assert cfg.data_dir == "/custom/data"


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
