"""Temporal code intelligence — git history scoped by flowmap's index."""

from flowmap.history.git_ops import CommitInfo, FileHistory, get_file_at_commit, get_file_history, pickaxe_search
from flowmap.history.structural_diff import SymbolChange, structural_diff
from flowmap.history.timeline import Timeline, TimelineEntry, build_timeline

__all__ = [
    "CommitInfo",
    "FileHistory",
    "SymbolChange",
    "Timeline",
    "TimelineEntry",
    "build_timeline",
    "get_file_at_commit",
    "get_file_history",
    "pickaxe_search",
    "structural_diff",
]
