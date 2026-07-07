"""FlowMap MCP Transport - Exposes FlowMap tools via MCP protocol."""

import logging
import sys
from pathlib import Path

# Add parent path for flowmap imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mcp.server import FastMCP

from .server import FlowMapMCPServer

logger = logging.getLogger(__name__)


def create_flowmap_mcp_server() -> FastMCP:
    """Create an MCP server wrapping FlowMap tools."""
    
    mcp = FastMCP("FlowMap Code Intelligence")
    flowmap = FlowMapMCPServer()
    
    @mcp.tool()
    def flowmap_search(
        query: str,
        mode: str = "auto",
        repo: str = None,
        regex: bool = False,
        limit: int = 10
    ) -> str:
        """Search code across all indexed repositories.
        
        Args:
            query: Search query (natural language or keywords)
            mode: Search mode - "semantic" for AI search, "keyword" for exact match, "symbol" for code symbols, "auto" picks best
            repo: Filter to specific repository (optional)
            regex: Enable regex in keyword mode
            limit: Maximum number of results
        """
        results = flowmap.search(query, mode=mode, repo=repo, regex=regex, limit=limit)
        if not results:
            return "No results found."
        
        output = []
        for i, r in enumerate(results[:limit], 1):
            loc = f"{r.repo}/{r.file}"
            if r.start_line:
                loc += f":{r.start_line}"
            output.append(f"[{i}] {loc}\n    {r.text[:200]}")
        
        return f"Found {len(results)} results:\n\n" + "\n\n".join(output)
    
    @mcp.tool()
    def flowmap_cat(
        repo: str,
        file: str,
        lines: str = None,
        symbol: str = None
    ) -> str:
        """Read a file from any indexed repository.
        
        Args:
            repo: Repository name
            file: File path within the repository
            lines: Line range like "100-150" (optional)
            symbol: Symbol name to find like "ClassName.method" (optional)
        """
        try:
            content = flowmap.cat(repo, file, lines=lines, symbol=symbol)
            return content
        except FileNotFoundError as e:
            return f"Error: {e}"
    
    @mcp.tool()
    def flowmap_history(
        query: str,
        repo: str = None,
        limit: int = 10
    ) -> str:
        """Get commit history related to a query.
        
        Args:
            query: Search query for commit messages
            repo: Filter to specific repository (optional)
            limit: Maximum number of commits to return
        """
        commits = flowmap.history(query, repo=repo, limit=limit)
        if not commits:
            return "No commits found."
        
        output = []
        for c in commits:
            output.append(f"[{c.sha[:8]}] {c.message}")
        
        return f"Found {len(commits)} commits:\n\n" + "\n".join(output)
    
    @mcp.tool()
    def flowmap_repos() -> str:
        """List all indexed repositories with their chunk counts."""
        repos = flowmap.repos()
        if not repos:
            return "No repositories indexed."
        
        output = []
        for r in repos:
            output.append(f"- {r['name']} ({r.get('chunk_count', '?')} chunks)")
        
        return "Indexed repositories:\n" + "\n".join(output)
    
    @mcp.tool()
    def flowmap_symbols(
        repo: str,
        limit: int = 30
    ) -> str:
        """List symbols (classes, functions, etc.) in a repository.
        
        Args:
            repo: Repository name
            limit: Maximum number of symbols to return
        """
        results = flowmap.symbols(repo, limit=limit)
        if not results:
            return "No symbols found."
        
        output = []
        for r in results:
            output.append(f"[{r.symbol_name}] {r.repo}/{r.file}:{r.start_line}")
        
        return f"Found {len(results)} symbols:\n\n" + "\n".join(output)
    
    @mcp.tool()
    def flowmap_map(repo: str = None) -> str:
        """Show repository structure.
        
        Args:
            repo: Specific repository to show (optional, shows all if not provided)
        """
        return flowmap.map(repo)
    
    return mcp


def main():
    """Run the FlowMap MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlowMap MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3001, help="Port to listen on")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting FlowMap MCP Server on {args.host}:{args.port}")
    
    mcp = create_flowmap_mcp_server()
    
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
