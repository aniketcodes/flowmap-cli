"""FlowMap MCP Server - Exposes FlowMap code intelligence as MCP tools."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "flowmap-cli"))

from flowmap.config import load_config
from flowmap.embeddings import create_backend
from flowmap.store import VectorStore
from flowmap.search.hybrid import hybrid_search, HybridResult
from flowmap.search.ripgrep import rg_search
from flowmap.history.timeline import build_timeline
from flowmap.state import StateDB

logger = logging.getLogger(__name__)

DOC_PATTERNS = (".md", ".txt", ".rst", "CLAUDE.md", "README")


@dataclass
class SearchResult:
    """Structured search result."""
    repo: str
    file: str
    start_line: int
    end_line: int
    text: str
    score: float
    symbol_name: str = ""
    chunk_type: str = ""
    signature: str = ""


@dataclass
class CommitInfo:
    """Structured commit info."""
    sha: str
    message: str
    author: str = ""
    date: str = ""
    repo: str = ""


class FlowMapMCPServer:
    """MCP server exposing FlowMap tools — mirrors the CLI capabilities."""

    TOOL_NAMES = [
        "flowmap_search",
        "flowmap_cat",
        "flowmap_history",
        "flowmap_repos",
        "flowmap_symbols",
        "flowmap_map",
    ]

    def __init__(self, config_path: str = None):
        self._config = load_config(config_path) if config_path else load_config()
        self._backend = create_backend(
            self._config.embedding.backend,
            self._config.embedding.model,
            self._config.embedding.ollama_url,
        )
        self._store = VectorStore(
            db_path=str(self._config.lancedb_path),
            vector_dims=self._backend.dims(),
        )
        self._state = StateDB(self._config.db_path)
        self._tools = self.TOOL_NAMES.copy()
        logger.info("flowmap_mcp_init tools=%s repos=%s", self._tools, list(self._config.repo_paths().keys()))

    def list_tools(self) -> list[dict]:
        """Return MCP-compliant tool definitions."""
        return [
            {
                "name": "flowmap_search",
                "description": "Search code across all indexed repositories. Use semantic for AI-powered search, keyword for exact matches, symbol for code symbols.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (natural language or keywords)"},
                        "mode": {"type": "string", "enum": ["auto", "semantic", "keyword", "symbol"], "description": "Search mode"},
                        "repo": {"type": "string", "description": "Filter to specific repository"},
                        "regex": {"type": "boolean", "description": "Enable regex in keyword mode"},
                        "limit": {"type": "integer", "description": "Maximum number of results"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "flowmap_cat",
                "description": "Read a file from any indexed repository. Use lines for line ranges, symbol for specific functions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Repository name"},
                        "file": {"type": "string", "description": "File path within the repository"},
                        "lines": {"type": "string", "description": "Line range like '100-150'"},
                        "symbol": {"type": "string", "description": "Symbol name like 'ClassName.method'"}
                    },
                    "required": ["repo", "file"]
                }
            },
            {
                "name": "flowmap_history",
                "description": "Get commit history related to a query. Shows recent changes to code.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for commit messages"},
                        "repo": {"type": "string", "description": "Filter to specific repository"},
                        "limit": {"type": "integer", "description": "Maximum number of commits to return"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "flowmap_repos",
                "description": "List all indexed repositories with their chunk counts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "flowmap_symbols",
                "description": "List symbols (classes, functions, etc.) defined in a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Repository name"},
                        "limit": {"type": "integer", "description": "Maximum number of symbols to return"}
                    },
                    "required": ["repo"]
                }
            },
            {
                "name": "flowmap_map",
                "description": "Show repository structure and file organization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Specific repository to show (optional)"}
                    },
                    "required": []
                }
            }
        ]

    # ── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        mode: str = "auto",
        repo: str = None,
        regex: bool = False,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search code across all indexed repos.

        mode: "semantic" | "keyword" | "symbol" | "auto" (default picks best)
        repo: filter to a specific repo
        regex: enable regex in keyword mode
        limit: max results
        """
        if not query or not query.strip():
            return []

        try:
            if mode == "keyword":
                # Ripgrep-only: fast exact match
                repo_paths = self._config.repo_paths()
                if repo:
                    repo_paths = {k: v for k, v in repo_paths.items() if k == repo}
                rg_results = rg_search(query, repo_paths, limit=limit, regex=regex)
                return [
                    SearchResult(
                        repo=r.repo, file=r.file,
                        start_line=r.line, end_line=r.line,
                        text=r.text, score=1.0,
                    )
                    for r in rg_results[:limit]
                ]

            if mode == "symbol":
                results = self._store.search_symbol(query, repo_filter=repo, limit=limit)
                return [self._to_search_result(r) for r in results]

            # auto / semantic — full hybrid pipeline
            results = hybrid_search(
                query=query,
                repo_paths=self._config.repo_paths(),
                embedding_backend=self._backend,
                store=self._store,
                limit=limit,
                repo_filter=repo,
                regex=regex,
            )

            # Deprioritize docs
            scored = []
            for r in results:
                penalty = 0.3 if any(p in r.file for p in DOC_PATTERNS) else 1.0
                scored.append((r, r.score * penalty))
            scored.sort(key=lambda x: x[1], reverse=True)

            logger.info("flowmap_search query=%s mode=%s repo=%s results=%d", query[:50], mode, repo, len(scored))
            return [self._to_search_result(r) for r, _ in scored[:limit]]

        except Exception as e:
            logger.error("flowmap_search_error query=%s error=%s", query[:50], e, exc_info=True)
            return []

    # ── cat ─────────────────────────────────────────────────────────────

    def cat(self, repo: str, file: str, lines: str = None, symbol: str = None) -> str:
        """Read a file from any indexed repo.

        lines: "100-150" line range (optional)
        symbol: "ClassName.method" to find and return that symbol's code (optional)
        """
        repo_paths = self._config.repo_paths()
        if repo not in repo_paths:
            raise FileNotFoundError(f"Repository '{repo}' not found")

        file_path = Path(repo_paths[repo]) / file
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file}' not found in {repo}")

        content = file_path.read_text()

        if symbol:
            # Find the symbol in the file content
            lines_list = content.split("\n")
            for i, line in enumerate(lines_list):
                if symbol in line:
                    # Return surrounding context (20 lines before/after)
                    start = max(0, i - 20)
                    end = min(len(lines_list), i + 40)
                    return "\n".join(
                        f"{j+1:4d}: {lines_list[j]}"
                        for j in range(start, end)
                    )
            return f"Symbol '{symbol}' not found in {repo}/{file}"

        if lines:
            parts = lines.split("-")
            try:
                start = int(parts[0]) - 1
                end = int(parts[1]) if len(parts) > 1 else start + 1
                lines_list = content.split("\n")
                return "\n".join(
                    f"{i+1:4d}: {lines_list[i]}"
                    for i in range(max(0, start), min(len(lines_list), end))
                )
            except (ValueError, IndexError):
                pass

        return content

    # ── history ─────────────────────────────────────────────────────────

    def history(self, query: str, repo: str = None, limit: int = 10) -> list[CommitInfo]:
        """Get commit history related to a query."""
        if not query or not query.strip():
            return []

        try:
            repo_paths = self._config.repo_paths()
            if repo:
                repo_paths = {k: v for k, v in repo_paths.items() if k == repo}

            timeline = build_timeline(
                query=query,
                repo_paths=repo_paths,
                store=self._store,
                embedding_backend=self._backend,
            )

            # Deduplicate by SHA — TimelineEntry is per (commit, file) pair
            seen_shas = set()
            commits = []
            for entry in timeline.entries[:limit]:
                if entry.commit.sha in seen_shas:
                    continue
                seen_shas.add(entry.commit.sha)
                commits.append(CommitInfo(
                    sha=entry.commit.sha,
                    message=entry.commit.message,
                    author=entry.commit.author,
                    date=entry.commit.date,
                    repo=entry.repo,
                ))
            logger.info("flowmap_history query=%s repo=%s commits=%d", query[:50], repo, len(commits))
            return commits
        except Exception as e:
            logger.error("flowmap_history_error query=%s error=%s", query[:50], e, exc_info=True)
            return []

    # ── repos ───────────────────────────────────────────────────────────

    def repos(self) -> list[dict]:
        """List all indexed repositories."""
        return self._state.list_repos()

    # ── symbols ─────────────────────────────────────────────────────────

    def symbols(self, repo: str, limit: int = 30) -> list[SearchResult]:
        """List symbols in a repo."""
        results = self._store.search_symbol("*", repo_filter=repo, limit=limit)
        return [self._to_search_result(r) for r in results]

    # ── map ─────────────────────────────────────────────────────────────

    def map(self, repo: str = None) -> str:
        """Show repo structure."""
        repos = self.repos()
        if repo:
            filtered = [r for r in repos if r["name"] == repo]
            if not filtered:
                return f"Repository '{repo}' not found. Available: {', '.join(r['name'] for r in repos)}"
            repos = filtered
        return "\n".join(f"- {r['name']} ({r.get('chunk_count', '?')} chunks)" for r in repos)

    # ── dispatch ────────────────────────────────────────────────────────

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        handlers = {
            "flowmap_search": lambda a: self.search(
                a.get("query", ""),
                mode=a.get("mode", "auto"),
                repo=a.get("repo"),
                regex=a.get("regex", False),
                limit=a.get("limit", 10),
            ),
            "flowmap_cat": lambda a: self.cat(
                a.get("repo", ""),
                a.get("file", ""),
                lines=a.get("lines"),
                symbol=a.get("symbol"),
            ),
            "flowmap_history": lambda a: self.history(
                a.get("query", ""),
                repo=a.get("repo"),
                limit=a.get("limit", 10),
            ),
            "flowmap_repos": lambda a: self.repos(),
            "flowmap_symbols": lambda a: self.symbols(
                a.get("repo", ""),
                limit=a.get("limit", 30),
            ),
            "flowmap_map": lambda a: self.map(a.get("repo")),
        }

        if name not in handlers:
            raise ValueError(f"Unknown tool: {name}")

        return handlers[name](args)

    # ── helpers ─────────────────────────────────────────────────────────

    def _to_search_result(self, r) -> SearchResult:
        return SearchResult(
            repo=r.repo, file=r.file,
            start_line=r.start_line, end_line=r.end_line,
            text=r.text, score=r.score,
            symbol_name=getattr(r, "symbol_name", ""),
            chunk_type=getattr(r, "chunk_type", ""),
            signature=getattr(r, "signature", ""),
        )
