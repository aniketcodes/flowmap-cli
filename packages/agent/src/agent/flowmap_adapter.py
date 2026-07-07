"""FlowMap Adapter — Wraps FlowMapMCPServer with MCPAdapter interface."""

import logging
from typing import Any

from .adapter import MCPAdapter
from .server import FlowMapMCPServer, SearchResult, CommitInfo

logger = logging.getLogger(__name__)


class FlowMapAdapter(MCPAdapter):
    """Adapter for FlowMap code intelligence MCP server.

    Wraps FlowMapMCPServer and exposes it through the standard MCPAdapter
    interface.
    """

    def __init__(self, server: FlowMapMCPServer = None):
        self._server = server or FlowMapMCPServer()

    def name(self) -> str:
        return "flowmap"

    def list_tools(self) -> list[dict]:
        return self._server.list_tools()

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route tool call to appropriate server method."""
        # Remove prefix if present
        method_name = name.removeprefix("flowmap_")

        # Map tool names to handlers
        handlers = {
            "search": self._handle_search,
            "cat": self._handle_cat,
            "history": self._handle_history,
            "repos": self._handle_repos,
            "symbols": self._handle_symbols,
            "map": self._handle_map,
        }

        handler = handlers.get(method_name)
        if not handler:
            return f"Unknown tool: {name}"

        try:
            return handler(args)
        except Exception as e:
            logger.error("FlowMap tool error: %s %s", method_name, e, exc_info=True)
            return f"Error: {e}"

    def _handle_search(self, args: dict) -> str:
        results = self._server.search(
            query=args.get("query", ""),
            mode=args.get("mode", "auto"),
            repo=args.get("repo"),
            regex=args.get("regex", False),
            limit=int(args.get("limit", 10)),
        )
        return self._format_search_results(results)

    def _handle_cat(self, args: dict) -> str:
        content = self._server.cat(
            repo=args.get("repo", ""),
            file=args.get("file", ""),
            lines=args.get("lines"),
            symbol=args.get("symbol"),
        )
        return content[:3000]  # Truncate for LLM context

    def _handle_history(self, args: dict) -> str:
        commits = self._server.history(
            query=args.get("query", ""),
            repo=args.get("repo"),
            limit=int(args.get("limit", 10)),
        )
        return self._format_commits(commits)

    def _handle_repos(self, args: dict) -> str:
        repos = self._server.repos()
        return "\n".join(f"- {r['name']} ({r.get('chunk_count', '?')} chunks)" for r in repos)

    def _handle_symbols(self, args: dict) -> str:
        results = self._server.symbols(
            repo=args.get("repo", ""),
            limit=int(args.get("limit", 30)),
        )
        return self._format_search_results(results)

    def _handle_map(self, args: dict) -> str:
        return self._server.map(args.get("repo"))

    def _format_search_results(self, results: list[SearchResult]) -> str:
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results):
            loc = f"{r.repo}/{r.file}"
            if r.start_line:
                loc += f":{r.start_line}"
            formatted.append(f"[{i+1}] {loc}\n    {r.text[:300]}")

        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)

    def _format_commits(self, commits: list[CommitInfo]) -> str:
        if not commits:
            return "No commits found."

        formatted = []
        for c in commits:
            formatted.append(f"[{c.sha[:8]}] {c.date} by {c.author}: {c.message}")

        return f"Found {len(commits)} commits:\n\n" + "\n".join(formatted)
