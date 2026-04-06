"""Service layer — testable business logic extracted from CLI commands."""

from flowmap.services.file_resolver import ResolvedFile, resolve_file
from flowmap.services.indexing import IndexResult, embed_chunks, run_index
from flowmap.services.symbol_lookup import SymbolMatch, get_symbol_suggestions, resolve_symbol

__all__ = [
    "IndexResult",
    "ResolvedFile",
    "SymbolMatch",
    "embed_chunks",
    "get_symbol_suggestions",
    "resolve_file",
    "resolve_symbol",
    "run_index",
]
