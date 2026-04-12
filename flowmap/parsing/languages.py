"""Tree-sitter language registry — maps file extensions to grammars and chunking strategies."""

from __future__ import annotations

import logging
from functools import lru_cache

import tree_sitter as ts

log = logging.getLogger(__name__)

# (grammar_loader_module, loader_function_name, chunking_strategy)
# chunking_strategy: "code" = AST function/class extraction, "config" = top-level keys, "fallback" = line-based
_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Code files
    ".py":   ("tree_sitter_python",     "language",            "code"),
    ".js":   ("tree_sitter_javascript",  "language",            "code"),
    ".ts":   ("tree_sitter_typescript",  "language_typescript",  "code"),
    ".tsx":  ("tree_sitter_typescript",  "language_tsx",         "code"),
    ".jsx":  ("tree_sitter_javascript",  "language",            "code"),
    ".go":   ("tree_sitter_go",          "language",            "code"),
    ".java": ("tree_sitter_java",        "language",            "code"),
    ".swift": ("tree_sitter_swift",      "language",            "code"),
    # Config files
    ".json": ("tree_sitter_json",        "language",            "config"),
    ".yaml": ("tree_sitter_yaml",        "language",            "config"),
    ".yml":  ("tree_sitter_yaml",        "language",            "config"),
    # Fallback (no AST parsing)
    ".sh":   (None, None, "fallback"),
    ".bash": (None, None, "fallback"),
    ".sql":  (None, None, "fallback"),
    ".md":   (None, None, "fallback"),
    ".txt":  (None, None, "fallback"),
    ".rst":  (None, None, "fallback"),
    ".toml": (None, None, "fallback"),
    ".tf":   (None, None, "fallback"),
    ".hcl":  (None, None, "fallback"),
    ".proto": (None, None, "fallback"),
    ".graphql": (None, None, "fallback"),
}

# Filenames without extensions that we support (use fallback chunking)
SUPPORTED_FILENAMES = {"Dockerfile", "Makefile", "Jenkinsfile", "Vagrantfile"}


# Code extensions that lack a tree-sitter grammar — users should know these get fallback chunking
_CODE_EXTENSIONS_WITHOUT_GRAMMAR = {
    ".rs", ".c", ".cpp", ".h", ".hpp", ".kt", ".rb", ".php", ".cs",
}

_WARNED_EXTENSIONS: set[str] = set()


@lru_cache(maxsize=32)
def _get_language(extension: str) -> tuple[ts.Language | None, str]:
    """Cached language loader. Language objects are immutable and thread-safe.

    Returns (language, strategy) or (None, "fallback").
    """
    entry = _REGISTRY.get(extension)
    if entry is None:
        if extension not in _WARNED_EXTENSIONS:
            _WARNED_EXTENSIONS.add(extension)
            if extension in _CODE_EXTENSIONS_WITHOUT_GRAMMAR:
                log.info("No tree-sitter grammar for %s — using line-based fallback chunking", extension)
            else:
                log.debug("No tree-sitter grammar for %s, using fallback chunking", extension)
        return None, "fallback"

    module_name, func_name, strategy = entry
    if module_name is None:
        return None, strategy

    try:
        mod = __import__(module_name)
        lang_fn = getattr(mod, func_name)
        lang = ts.Language(lang_fn())
        return lang, strategy
    except Exception as e:
        log.warning("Failed to load tree-sitter grammar for %s: %s (using fallback)", extension, e)
        return None, "fallback"


def get_parser(extension: str) -> tuple[ts.Parser | None, str]:
    """Return (parser, strategy) for a file extension.

    Creates a fresh Parser each call — Parser objects hold mutable state
    and must not be shared across threads. The underlying Language object
    is cached and immutable.
    """
    lang, strategy = _get_language(extension)
    if lang is None:
        return None, strategy
    return ts.Parser(lang), strategy


def get_language_name(extension: str) -> str:
    """Map extension to a normalized language name."""
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".go": "go",
        ".java": "java",
        ".swift": "swift",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "shell",
        ".bash": "shell",
        ".sql": "sql",
        ".md": "markdown",
        ".toml": "toml",
    }
    return mapping.get(extension, extension.lstrip("."))
