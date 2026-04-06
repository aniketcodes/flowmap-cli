"""FlowMap configuration — loaded from ~/.flowmap/config.yaml with env var fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_DIR = Path.home() / ".flowmap"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_DATA_DIR = DEFAULT_CONFIG_DIR / "data"

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "vendor", "dist", "build", ".next", ".nuxt",
    "target", ".idea", ".vscode", ".cursor",
    "coverage", ".tox", "egg-info",
    ".server", ".cache", ".npm", ".yarn",
    "site-packages", "lib", "lib64",  # virtualenv dirs (only affects non-git fallback walker)
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock",
}

SUPPORTED_EXTENSIONS = {
    # Code
    ".py", ".ts", ".js", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".cs", ".swift",
    # Config / infra
    ".yaml", ".yml", ".toml", ".json",
    ".sh", ".bash",
    ".sql", ".graphql", ".proto",
    ".tf", ".hcl",
    # Documentation
    ".md", ".txt", ".rst",
}

# Filenames matched without extension (e.g. Dockerfile, Makefile)
SUPPORTED_FILENAMES = {
    "Dockerfile", "Makefile", "Jenkinsfile", "Vagrantfile",
}

MAX_FILE_SIZE = 512 * 1024  # 512 KB


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RepoConfig:
    name: str
    path: str

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass
class EmbeddingConfig:
    backend: str = "ollama"
    model: str = "qwen3-embedding:0.6b"
    ollama_url: str = "http://localhost:11434"


@dataclass
class RerankingConfig:
    enabled: bool = False  # Disabled by default — adds ~10s latency. Enable with --rerank flag.
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class FlowmapConfig:
    repos: list[RepoConfig] = field(default_factory=list)
    data_dir: str = str(DEFAULT_DATA_DIR)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_path / "flowmap.db"

    @property
    def lancedb_path(self) -> Path:
        return self.data_path / "lancedb"

    def repo_paths(self) -> dict[str, str]:
        """Return {repo_name: resolved_path_str} for all configured repos."""
        return {r.name: str(r.resolved_path()) for r in self.repos}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_config_dict(raw: dict) -> FlowmapConfig:
    repos = [
        RepoConfig(name=r["name"], path=r["path"])
        for r in raw.get("repos", [])
    ]

    emb_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(
        backend=emb_raw.get("backend", EmbeddingConfig.backend),
        model=emb_raw.get("model", EmbeddingConfig.model),
        ollama_url=emb_raw.get("ollama_url", EmbeddingConfig.ollama_url),
    )

    rer_raw = raw.get("reranking", {})
    reranking = RerankingConfig(
        enabled=bool(rer_raw.get("enabled", RerankingConfig.enabled)),
        model=rer_raw.get("model", RerankingConfig.model),
    )

    return FlowmapConfig(
        repos=repos,
        data_dir=raw.get("data_dir", str(DEFAULT_DATA_DIR)),
        embedding=embedding,
        reranking=reranking,
    )


def load_config(config_path: Path | None = None) -> FlowmapConfig:
    """Load config from YAML file. Falls back to defaults if file missing."""
    path = config_path or DEFAULT_CONFIG_PATH

    if path.exists():
        with open(path, "r") as f:
            try:
                raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {path}: {e}")
        return _parse_config_dict(raw)

    # Fall back to env vars for backward compat
    return _config_from_env()


def _config_from_env() -> FlowmapConfig:
    """Build config from environment variables (backward compat)."""
    cfg = FlowmapConfig()

    if os.getenv("FLOWMAP_DATA_DIR"):
        cfg.data_dir = os.environ["FLOWMAP_DATA_DIR"]

    backend = os.getenv("FLOWMAP_EMBEDDING_BACKEND")
    if backend:
        cfg.embedding.backend = backend

    model = os.getenv("FLOWMAP_EMBEDDING_MODEL")
    if model:
        cfg.embedding.model = model

    ollama_url = os.getenv("FLOWMAP_OLLAMA_URL")
    if ollama_url:
        cfg.embedding.ollama_url = ollama_url

    return cfg


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_TEMPLATE = """\
# FlowMap configuration
# Docs: https://github.com/aniket-agi/flowmap-cli

repos:
  # - name: my-service
  #   path: /Users/you/code/my-service

data_dir: ~/.flowmap/data

embedding:
  backend: ollama                        # ollama | sentence-transformers
  model: qwen3-embedding:0.6b           # model name (backend-specific)
  ollama_url: http://localhost:11434
  # To use sentence-transformers instead (requires pip install flowmap[local-embeddings]):
  # backend: sentence-transformers
  # model: nomic-ai/CodeRankEmbed

reranking:
  enabled: false                        # adds ~10s latency (loads PyTorch). Use --rerank flag for quality-critical queries.
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""


def write_default_config(config_path: Path | None = None, force: bool = False) -> Path:
    """Write a starter config.yaml. Refuses to overwrite unless force=True.

    With force=True, preserves the existing repos list.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists() and not force:
        raise FileExistsError(f"Config already exists: {path}")

    # Preserve existing repos if force-overwriting
    existing_repos = []
    if force and path.exists():
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
            existing_repos = raw.get("repos") or []
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)

    if existing_repos:
        # Write template then inject repos
        raw = yaml.safe_load(DEFAULT_CONFIG_TEMPLATE) or {}
        raw["repos"] = existing_repos
        with open(path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    else:
        path.write_text(DEFAULT_CONFIG_TEMPLATE)

    return path


def add_repo_to_config(repo_path: str, name: str | None = None, config_path: Path | None = None) -> RepoConfig:
    """Add a repo to config.yaml. Uses yaml.dump (comments are not preserved)."""
    path = config_path or DEFAULT_CONFIG_PATH

    resolved = Path(repo_path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a directory: {resolved}")

    repo_name = name or resolved.name

    # Create config if it doesn't exist
    if not path.exists():
        write_default_config(path)

    # Load, modify, write back
    with open(path, "r") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    repos = raw.get("repos")
    if repos is None:
        repos = []
        raw["repos"] = repos

    # Check for duplicates
    for r in repos:
        if r.get("name") == repo_name:
            raise ValueError(f"Repo '{repo_name}' already in config")

    repos.append({"name": repo_name, "path": str(resolved)})

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return RepoConfig(name=repo_name, path=str(resolved))


def remove_repo_from_config(repo_name: str, config_path: Path | None = None) -> bool:
    """Remove a repo from config.yaml by name. Returns True if removed, False if not found."""
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return False

    with open(path, "r") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            return False

    repos = raw.get("repos") or []
    original_len = len(repos)
    repos = [r for r in repos if r.get("name") != repo_name]

    if len(repos) == original_len:
        return False

    raw["repos"] = repos
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return True


# ---------------------------------------------------------------------------
# .flowmapignore
# ---------------------------------------------------------------------------

def load_ignore_patterns(repo_path: str | Path) -> list[str]:
    """Load .flowmapignore patterns from a repo root. Returns empty list if no file."""
    ignore_file = Path(repo_path) / ".flowmapignore"
    if not ignore_file.exists():
        return []
    return [
        line.strip()
        for line in ignore_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def should_skip_path(rel_path: str, ignore_spec) -> bool:
    """Check if a relative path matches the ignore spec (pathspec.PathSpec)."""
    if ignore_spec is None:
        return False
    return ignore_spec.match_file(rel_path)
