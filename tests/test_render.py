"""Tests for render.py — output formatting functions."""

import json
from dataclasses import dataclass, field

from flowmap.render import (
    render_hybrid_results,
    render_keyword_results,
    render_map,
    render_semantic_results,
    render_symbol_results,
    render_symbols,
    render_timeline,
)


@dataclass
class _MockHybridResult:
    repo: str = "r1"
    file: str = "a.py"
    start_line: int = 1
    end_line: int = 10
    text: str = "def foo(): pass"
    score: float = 0.9
    rrf_score: float = 0.5
    symbol_name: str = "foo"
    signature: str = "def foo():"
    chunk_type: str = "function"
    parent_symbol: str = ""
    parent_signature: str = ""
    language: str = "python"
    match_type: str = "semantic"
    rerank_score: float = 0.0
    sources: list = field(default_factory=lambda: ["semantic"])


@dataclass
class _MockSearchResult:
    repo: str = "r1"
    file: str = "a.py"
    start_line: int = 1
    end_line: int = 10
    text: str = "def foo(): pass"
    score: float = 0.9
    symbol_name: str = "foo"
    signature: str = "def foo():"
    chunk_type: str = "function"
    parent_symbol: str = ""
    parent_signature: str = ""
    language: str = "python"


@dataclass
class _MockRgResult:
    repo: str = "r1"
    file: str = "a.py"
    line: int = 5
    text: str = "hello world"


class TestRenderHybrid:
    def test_json(self):
        results = [_MockHybridResult()]
        out = render_hybrid_results(results, "test", "json")
        data = json.loads(out)
        assert data["query"] == "test"
        assert len(data["results"]) == 1
        assert data["results"][0]["symbol_name"] == "foo"

    def test_text(self):
        results = [_MockHybridResult()]
        out = render_hybrid_results(results, "test", "text")
        assert "foo" in out
        assert "r1/a.py" in out


class TestRenderSemantic:
    def test_json(self):
        results = [_MockSearchResult()]
        out = render_semantic_results(results, "test", "json")
        data = json.loads(out)
        assert data["mode"] == "semantic"

    def test_text(self):
        results = [_MockSearchResult()]
        out = render_semantic_results(results, "test", "text")
        assert "score:" in out


class TestRenderSymbolResults:
    def test_json(self):
        results = [_MockSearchResult()]
        out = render_symbol_results(results, "test", "json")
        data = json.loads(out)
        assert data["mode"] == "symbol"


class TestRenderKeyword:
    def test_text(self):
        results = [_MockRgResult()]
        out = render_keyword_results(results)
        assert "hello world" in out


class TestRenderMap:
    def test_json(self):
        repos = [{"name": "r1", "files": 5, "languages": {"python": 3}, "classes": [], "functions": []}]
        out = render_map(repos, "json")
        data = json.loads(out)
        assert data["repos"][0]["name"] == "r1"

    def test_text(self):
        repos = [{"name": "r1", "files": 5, "languages": {"python": 3}, "classes": [], "functions": []}]
        out = render_map(repos, "text")
        assert "r1" in out
        assert "5 files" in out


class TestRenderSymbols:
    def test_json(self):
        rows = [{"symbol_name": "foo", "chunk_type": "function", "file": "a.py", "repo": "r1",
                 "start_line": 1, "signature": "def foo():", "language": "python"}]
        out = render_symbols(rows, "foo", "json")
        data = json.loads(out)
        assert data["symbols"][0]["symbol_name"] == "foo"

    def test_text(self):
        rows = [{"symbol_name": "foo", "chunk_type": "function", "file": "a.py", "repo": "r1",
                 "start_line": 1, "signature": "def foo():", "language": "python"}]
        out = render_symbols(rows, "foo", "text")
        assert "foo" in out


class TestRenderTimeline:
    def test_json(self):
        @dataclass
        class _Entry:
            commit: object
            file: str = "a.py"
            repo: str = "r1"
            changes: list = field(default_factory=list)
            relevance: str = "direct"

        @dataclass
        class _Commit:
            sha: str = "abc1234"
            author: str = "dev"
            date: str = "2026-01-15"
            message: str = "fix bug"

        @dataclass
        class _Timeline:
            query: str = "foo"
            entries: list = field(default_factory=list)
            scoped_files: list = field(default_factory=list)
            scoped_symbols: list = field(default_factory=list)

        tl = _Timeline(entries=[_Entry(commit=_Commit())])
        out = render_timeline(tl, "json")
        data = json.loads(out)
        assert data["query"] == "foo"
        assert data["entries"][0]["sha"] == "abc1234"

    def test_text(self):
        @dataclass
        class _Entry:
            commit: object
            file: str = "a.py"
            repo: str = "r1"
            changes: list = field(default_factory=list)
            relevance: str = "direct"

        @dataclass
        class _Commit:
            sha: str = "abc1234"
            author: str = "dev"
            date: str = "2026-01-15"
            message: str = "fix bug"

        @dataclass
        class _Timeline:
            query: str = "foo"
            entries: list = field(default_factory=list)
            scoped_files: list = field(default_factory=list)
            scoped_symbols: list = field(default_factory=list)

        tl = _Timeline(entries=[_Entry(commit=_Commit())])
        out = render_timeline(tl, "text")
        assert "abc1234" in out
        assert "fix bug" in out
