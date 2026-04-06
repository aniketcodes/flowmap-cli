"""Output formatting for CLI commands — text and JSON renderers."""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

def render_hybrid_results(results, query: str, fmt: str) -> str:
    """Format hybrid search results."""
    if fmt == "json":
        output = {
            "query": query,
            "mode": "hybrid",
            "results": [
                {
                    "symbol_name": r.symbol_name,
                    "signature": r.signature,
                    "parent_context": f"{r.parent_symbol}: {r.parent_signature}" if r.parent_symbol else "",
                    "file": r.file,
                    "repo": r.repo,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "chunk_type": r.chunk_type,
                    "language": r.language,
                    "match_type": r.match_type,
                    "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                    "rrf_score": round(r.rrf_score, 4),
                    "sources": r.sources,
                    "text": r.text,
                }
                for r in results
            ],
        }
        return json.dumps(output, indent=2)

    lines = []
    for i, r in enumerate(results, 1):
        loc = f"{r.repo}/{r.file}:{r.start_line}-{r.end_line}"
        sources = "+".join(r.sources)
        score_str = f"rerank: {r.rerank_score:.2f}" if r.rerank_score else f"rrf: {r.rrf_score:.4f}"
        lines.append(f"[{i}] {loc}  ({score_str}, via: {sources})")
        if r.symbol_name:
            lines.append(f"    {r.signature or r.symbol_name}")
        if r.parent_symbol:
            lines.append(f"    in: {r.parent_signature}")
        preview = r.text.replace("\n", " ")[:200]
        lines.append(f"    {preview}{'...' if len(r.text) > 200 else ''}")
        lines.append("")
    return "\n".join(lines)


def render_semantic_results(results, query: str, fmt: str) -> str:
    """Format semantic search results."""
    if fmt == "json":
        output = {
            "query": query,
            "mode": "semantic",
            "results": [
                {
                    "symbol_name": r.symbol_name,
                    "signature": r.signature,
                    "file": r.file,
                    "repo": r.repo,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": round(r.score, 4),
                    "text": r.text,
                }
                for r in results
            ],
        }
        return json.dumps(output, indent=2)

    lines = []
    for i, r in enumerate(results, 1):
        loc = f"{r.repo}/{r.file}:{r.start_line}-{r.end_line}"
        lines.append(f"[{i}] {loc}  (score: {r.score:.4f})")
        if r.symbol_name:
            lines.append(f"    {r.signature or r.symbol_name}")
        preview = r.text.replace("\n", " ")[:200]
        lines.append(f"    {preview}{'...' if len(r.text) > 200 else ''}")
        lines.append("")
    return "\n".join(lines)


def render_symbol_results(results, query: str, fmt: str) -> str:
    """Format symbol search results."""
    if fmt == "json":
        output = {
            "query": query,
            "mode": "symbol",
            "results": [
                {
                    "symbol_name": r.symbol_name,
                    "signature": r.signature,
                    "file": r.file,
                    "repo": r.repo,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "chunk_type": r.chunk_type,
                }
                for r in results
            ],
        }
        return json.dumps(output, indent=2)

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.repo}/{r.file}:{r.start_line}  {r.symbol_name}")
        if r.signature:
            lines.append(f"    {r.signature}")
    return "\n".join(lines)


def render_keyword_results(results) -> str:
    """Format keyword (ripgrep) search results."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.repo}/{r.file}:{r.line}  {r.text[:120]}{'...' if len(r.text) > 120 else ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbols command
# ---------------------------------------------------------------------------

def render_symbols(rows, query: str | None, fmt: str) -> str:
    """Format symbols listing."""
    if fmt == "json":
        results = [
            {
                "symbol_name": r.get("symbol_name", ""),
                "chunk_type": r.get("chunk_type", ""),
                "file": r.get("file", ""),
                "repo": r.get("repo", ""),
                "line": r.get("start_line", 0),
                "signature": r.get("signature", ""),
                "language": r.get("language", ""),
            }
            for r in rows
        ]
        return json.dumps({"query": query or "", "symbols": results}, indent=2)

    lines = []
    for r in rows:
        sym = r.get("symbol_name", "")
        kind_str = r.get("chunk_type", "")
        file = r.get("file", "")
        repo_name = r.get("repo", "")
        line = r.get("start_line", 0)
        sig = r.get("signature", "")
        lines.append(f"  {sym:<40} {kind_str:<10} {repo_name}/{file}:{line}")
        if sig and sig != sym:
            lines.append(f"    {sig[:80]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Map command
# ---------------------------------------------------------------------------

def render_map(output_repos: list[dict], fmt: str) -> str:
    """Format map output."""
    if fmt == "json":
        return json.dumps({"repos": output_repos}, indent=2)

    lines = []
    for repo_out in output_repos:
        lines.append(f"\n{repo_out['name']}  ({repo_out['files']} files)")
        langs = ", ".join(f"{k}: {v}" for k, v in sorted(repo_out["languages"].items(), key=lambda x: -x[1]))
        lines.append(f"  Languages: {langs}")

        if repo_out["classes"]:
            lines.append(f"  Classes ({len(repo_out['classes'])}):")
            for c in repo_out["classes"][:20]:
                methods_str = ", ".join(c["methods"][:5])
                if len(c["methods"]) > 5:
                    methods_str += f", ... (+{len(c['methods']) - 5})"
                lines.append(f"    {c['name']:<30} {c['file']}:{c['line']}")
                if methods_str:
                    lines.append(f"      methods: {methods_str}")

        if repo_out["functions"]:
            lines.append(f"  Functions ({len(repo_out['functions'])}):")
            for f in repo_out["functions"][:30]:
                sig = f["signature"][:60] if f["signature"] else f["name"]
                lines.append(f"    {sig:<60} {f['file']}:{f['line']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History / timeline
# ---------------------------------------------------------------------------

def render_timeline(timeline, fmt: str) -> str:
    """Format history timeline."""
    if fmt == "json":
        output = {
            "query": timeline.query,
            "scoped_files": timeline.scoped_files,
            "scoped_symbols": timeline.scoped_symbols,
            "entries": [
                {
                    "sha": e.commit.sha,
                    "author": e.commit.author,
                    "date": e.commit.date,
                    "message": e.commit.message,
                    "repo": e.repo,
                    "file": e.file,
                    "relevance": e.relevance,
                    "changes": [
                        {
                            "symbol": c.symbol_name,
                            "change_type": c.change_type,
                            "significance": c.significance,
                            "old_signature": c.old_signature,
                            "new_signature": c.new_signature,
                        }
                        for c in e.changes
                    ],
                }
                for e in timeline.entries
            ],
        }
        return json.dumps(output, indent=2)

    n_files = len(timeline.scoped_files)
    n_commits = len(timeline.entries)
    lines = [f'\nTimeline for "{timeline.query}" ({n_files} files, {n_commits} commits)\n']

    for entry in timeline.entries:
        sha_short = entry.commit.sha[:7]
        date_short = entry.commit.date[:10]
        lines.append(f"  {date_short}  {sha_short}  {entry.commit.author}  {entry.commit.message}")

        if entry.file:
            lines.append(f"    {entry.repo}/{entry.file}")

        for c in entry.changes:
            if c.change_type == "added":
                marker = "+"
            elif c.change_type == "removed":
                marker = "-"
            else:
                marker = "~"
            lines.append(f"      {marker} {c.symbol_name} ({c.change_type}, {c.significance})")
            if c.change_type == "signature_changed":
                lines.append(f"        was: {c.old_signature}")
                lines.append(f"        now: {c.new_signature}")

        lines.append("")
    return "\n".join(lines)
