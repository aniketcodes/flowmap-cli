"""Structural diff — compare file versions at the AST/symbol level using tree-sitter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from flowmap.parsing import chunk_file

log = logging.getLogger(__name__)


@dataclass
class SymbolChange:
    symbol_name: str
    change_type: str        # "added" | "removed" | "signature_changed" | "body_changed"
    old_signature: str      # empty for "added"
    new_signature: str      # empty for "removed"
    significance: str       # "major" | "minor" | "refactor"


def _build_symbol_map(content: str, extension: str) -> dict[str, dict]:
    """Parse content with tree-sitter and build {symbol_name: {signature, text}} map."""
    chunks = chunk_file("file" + extension, content, extension)
    symbols: dict[str, dict] = {}
    for chunk in chunks:
        if not chunk.symbol_name:
            continue
        # Use qualified name as key; if duplicates, later one wins (last definition)
        symbols[chunk.symbol_name] = {
            "signature": chunk.signature,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
        }
    return symbols


def _text_change_ratio(old_text: str, new_text: str) -> float:
    """Measure how much text changed (0.0 = identical, 1.0 = totally different).

    Uses SequenceMatcher for order-aware, duplicate-sensitive comparison.
    """
    if not old_text and not new_text:
        return 0.0
    if not old_text or not new_text:
        return 1.0
    return 1.0 - SequenceMatcher(None, old_text, new_text).ratio()


def structural_diff(
    old_content: str | None,
    new_content: str | None,
    extension: str,
) -> list[SymbolChange]:
    """Compare two file versions at the symbol level.

    Uses tree-sitter chunking to extract symbols from both versions,
    then compares: added, removed, signature_changed, body_changed.

    Returns empty list if the extension has no tree-sitter support
    or if both contents are None.
    """
    if old_content is None and new_content is None:
        return []

    # New file — all symbols are "added"
    if old_content is None:
        new_symbols = _build_symbol_map(new_content, extension)
        return [
            SymbolChange(
                symbol_name=name,
                change_type="added",
                old_signature="",
                new_signature=info["signature"],
                significance="major",
            )
            for name, info in new_symbols.items()
        ]

    # Deleted file — all symbols are "removed"
    if new_content is None:
        old_symbols = _build_symbol_map(old_content, extension)
        return [
            SymbolChange(
                symbol_name=name,
                change_type="removed",
                old_signature=info["signature"],
                new_signature="",
                significance="major",
            )
            for name, info in old_symbols.items()
        ]

    # Both versions exist — compare symbols
    old_symbols = _build_symbol_map(old_content, extension)
    new_symbols = _build_symbol_map(new_content, extension)

    if not old_symbols and not new_symbols:
        return []  # no parseable symbols (e.g., .md, unsupported extension)

    changes: list[SymbolChange] = []
    all_names = set(old_symbols.keys()) | set(new_symbols.keys())

    for name in sorted(all_names):
        old = old_symbols.get(name)
        new = new_symbols.get(name)

        if old is None and new is not None:
            changes.append(SymbolChange(
                symbol_name=name,
                change_type="added",
                old_signature="",
                new_signature=new["signature"],
                significance="major",
            ))
        elif old is not None and new is None:
            changes.append(SymbolChange(
                symbol_name=name,
                change_type="removed",
                old_signature=old["signature"],
                new_signature="",
                significance="major",
            ))
        elif old is not None and new is not None:
            if old["signature"] != new["signature"]:
                changes.append(SymbolChange(
                    symbol_name=name,
                    change_type="signature_changed",
                    old_signature=old["signature"],
                    new_signature=new["signature"],
                    significance="major",
                ))
            elif old["text"] != new["text"]:
                ratio = _text_change_ratio(old["text"], new["text"])
                changes.append(SymbolChange(
                    symbol_name=name,
                    change_type="body_changed",
                    old_signature=old["signature"],
                    new_signature=new["signature"],
                    significance="refactor" if ratio > 0.2 else "minor",
                ))
            # else: unchanged — skip

    return changes
