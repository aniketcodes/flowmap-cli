"""Symbol resolution — find symbols scoped to files with intelligent matching."""

from __future__ import annotations

from dataclasses import dataclass

from flowmap.store import SearchResult, VectorStore


@dataclass
class SymbolMatch:
    result: SearchResult
    source: str     # "file_scoped_dotted" | "file_scoped" | "global_fallback"


def resolve_symbol(
    symbol: str,
    repo_name: str,
    rel_file: str,
    store: VectorStore,
    profile: str = "default",
) -> SymbolMatch | None:
    """Resolve a symbol name within a file context.

    Strategy:
    1. Dotted names: "Class.method" → search base name with parent filter
    2. File-scoped search (exact → suffix → contains via store.search_symbol)
    3. Global fallback: repo-wide search, prefer results from matching file
    Returns None if not found.
    """
    # 1. Handle dotted names: "Class.method"
    if "." in symbol:
        parent_part, base_name = symbol.rsplit(".", 1)
        results = store.search_symbol(base_name, repo_filter=repo_name, file_filter=rel_file, limit=10, profile=profile)
        for r in results:
            if parent_part in (r.parent_symbol or r.symbol_name):
                return SymbolMatch(result=r, source="file_scoped_dotted")

    # 2. File-scoped search
    results = store.search_symbol(symbol, repo_filter=repo_name, file_filter=rel_file, limit=5, profile=profile)
    if results:
        return SymbolMatch(result=results[0], source="file_scoped")

    # 3. Global fallback with file preference (suffix path matching)
    results = store.search_symbol(symbol, repo_filter=repo_name, limit=20, profile=profile)
    for r in results:
        if r.file == rel_file or r.file.endswith("/" + rel_file):
            return SymbolMatch(result=r, source="global_fallback")

    return None


def get_symbol_suggestions(
    repo_name: str,
    rel_file: str,
    store: VectorStore,
    limit: int = 8,
    profile: str = "default",
) -> list[str]:
    """Get available symbol names in a file for error messages."""
    available = store.get_symbols(repo=repo_name, limit=200, profile=profile)
    file_syms = [s.get("symbol_name", "") for s in available if s.get("file") == rel_file]
    return file_syms[:limit]
