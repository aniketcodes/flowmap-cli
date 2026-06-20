"""FlowMap CLI — cross-repo code intelligence for LLMs."""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path

import click

from flowmap.config import (
    DEFAULT_CONFIG_PATH,
    FlowmapConfig,
    add_repo_to_config,
    load_config,
    remove_repo_from_config,
    write_default_config,
)


def _load_cfg(ctx: click.Context) -> FlowmapConfig:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    try:
        return load_config(config_path)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


log = logging.getLogger(__name__)


def _get_stored_dims(cfg: FlowmapConfig) -> int:
    """Read stored embedding dims for the active profile from StateDB, default 1024."""
    from flowmap.state import StateDB
    try:
        with StateDB(cfg.db_path) as state:
            stored = state.get_meta("embedding_dims", cfg.embedding.profile_name)
            return int(stored) if stored else 1024
    except (sqlite3.DatabaseError, ValueError, OSError) as e:
        log.warning("Could not read stored embedding dims: %s (defaulting to 1024)", e)
        return 1024


def _acquire_index_lock(lock_path: Path) -> int | None:
    """Acquire advisory file lock. Returns fd on success, None if locked."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        os.close(fd)
        return None
    return fd


def _release_index_lock(fd: int):
    """Release advisory file lock."""
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)


@click.group()
@click.version_option(package_name="flowmap", prog_name="flowmap")
@click.option("--config", "config_path", type=click.Path(exists=False), default=None, help="Path to config.yaml")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option("--json-log", is_flag=True, help="Structured JSON logging (for pipelines)")
@click.pass_context
def main(ctx, config_path, verbose, json_log):
    """FlowMap — cross-repo code intelligence CLI for LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config_path) if config_path else None
    if json_log:
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                })

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.DEBUG if verbose else logging.INFO)
    elif verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s %(levelname)s: %(message)s",
        )
    else:
        # Default: show warnings and errors on stderr (no --verbose needed)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# flowmap init
# ---------------------------------------------------------------------------

@main.command()
@click.option("--force", is_flag=True, help="Overwrite existing config")
@click.pass_context
def init(ctx, force):
    """Create a starter ~/.flowmap/config.yaml."""
    config_path = ctx.obj.get("config_path")
    try:
        path = write_default_config(config_path, force=force)
        click.echo(f"Created config: {path}")
        click.echo("Edit it to add your repos, then run: flowmap repos add /path/to/repo")
    except FileExistsError:
        path = config_path or DEFAULT_CONFIG_PATH
        click.echo(f"Config already exists: {path}")
        click.echo("Use --force to overwrite, or just run: flowmap repos add /path/to/repo")


# ---------------------------------------------------------------------------
# flowmap repos
# ---------------------------------------------------------------------------

@main.group()
def repos():
    """Manage indexed repositories."""


@repos.command("add")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--name", default=None, help="Alias for the repo (default: directory name)")
@click.pass_context
def repos_add(ctx, path, name):
    """Add a repository to the config."""
    config_path = ctx.obj.get("config_path")
    try:
        repo = add_repo_to_config(path, name=name, config_path=config_path)
        click.echo(f"Added: {repo.name} -> {repo.path}")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@repos.command("list")
@click.pass_context
def repos_list(ctx):
    """List all configured repositories and their index status."""
    cfg = _load_cfg(ctx)
    if not cfg.repos:
        click.echo("No repos configured. Run: flowmap repos add /path/to/repo")
        return

    from flowmap.state import StateDB
    from flowmap.store import VectorStore

    profile = cfg.embedding.profile_name
    # Live per-profile chunk counts; degrade to state-only if the store can't open
    # (keep this command robust — it was StateDB-only before).
    chunk_counts: dict[str, int] = {}
    try:
        with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
            chunk_counts = store.get_stats(profile=profile, known_repos=[r.name for r in cfg.repos]).get("repos", {})
    except Exception:
        pass

    with StateDB(cfg.db_path) as state:
        click.echo(f"Repos [profile: {profile}]:")
        for repo in cfg.repos:
            path_ok = repo.resolved_path().is_dir()
            idx = state.get_repo_index(repo.name, profile)
            if not path_ok:
                marker = click.style("!", fg="red")
                click.echo(f"  {marker} {repo.name:<28} path missing: {repo.path}")
            elif idx["sha"]:
                status = click.style("indexed", fg="green")
                chunks = chunk_counts.get(repo.name, 0)
                when = (idx["indexed_at"] or "")[:10]
                click.echo(f"    {repo.name:<28} {status}  {chunks:>6} chunks  {when}")
            else:
                status = click.style("not indexed", fg="yellow")
                click.echo(f"    {repo.name:<28} {status}")


@repos.command("paths")
@click.pass_context
def repos_paths(ctx):
    """Output all repo paths (one per line, for use with rg)."""
    cfg = _load_cfg(ctx)
    for repo in cfg.repos:
        click.echo(repo.resolved_path())


@repos.command("remove")
@click.argument("name")
@click.option("--keep-data", is_flag=True, help="Remove from config but keep index data")
@click.confirmation_option(prompt="Remove this repo from config and delete its index data?")
@click.pass_context
def repos_remove(ctx, name, keep_data):
    """Remove a repository from config and delete its index data."""
    config_path = ctx.obj.get("config_path")
    cfg = _load_cfg(ctx)

    # Check repo exists in config
    matching = [r for r in cfg.repos if r.name == name]
    if not matching:
        click.echo(f"Repo '{name}' not found in config. Run 'flowmap repos list' to see configured repos.", err=True)
        sys.exit(1)

    # Delete index data unless --keep-data
    if not keep_data:
        from flowmap.state import StateDB
        from flowmap.store import VectorStore
        try:
            with StateDB(cfg.db_path) as state, VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
                # Delete the repo's chunks from every profile table, not just default.
                for prof in store.list_profiles():
                    store.delete_by_repo(name, profile=prof)
                state.delete_repo(name)  # also clears per-profile staleness (atomic)
        except Exception as e:
            click.echo(f"Warning: could not clean index data: {e}", err=True)

    # Remove from config
    removed = remove_repo_from_config(name, config_path=config_path)
    if removed:
        if keep_data:
            click.echo(f"Removed '{name}' from config (index data kept).")
        else:
            click.echo(f"Removed '{name}' from config and deleted its index data.")
    else:
        click.echo(f"Could not remove '{name}' from config file.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# flowmap index
# ---------------------------------------------------------------------------

@main.command()
@click.option("--repo", default=None, help="Index a specific repo by name")
@click.option("--full", is_flag=True, help="Force full re-index (ignore incremental)")
@click.option("--dry-run", is_flag=True, help="Show what would be indexed without running")
@click.pass_context
def index(ctx, repo, full, dry_run):
    """Index repositories for search."""
    cfg = _load_cfg(ctx)
    if not cfg.repos:
        click.echo("No repos configured. Run: flowmap repos add /path/to/repo", err=True)
        sys.exit(1)

    targets = cfg.repos
    if repo:
        targets = [r for r in cfg.repos if r.name == repo]
        if not targets:
            resolved = Path(repo).expanduser().resolve()
            targets = [r for r in cfg.repos if r.resolved_path() == resolved]
        if not targets:
            click.echo(f"Repo '{repo}' not found in config. Run: flowmap repos list", err=True)
            sys.exit(1)

    if dry_run:
        from flowmap.config import SKIP_FILENAMES, SUPPORTED_EXTENSIONS
        from flowmap.indexer import _git_tracked_files
        from flowmap.reindex import get_git_status
        from flowmap.state import StateDB
        profile = cfg.embedding.profile_name
        click.echo(f"Dry run — showing what would be indexed [profile: {profile}]:\n")
        with StateDB(cfg.db_path) as state:
            for t in targets:
                resolved = t.resolved_path()
                if not resolved.is_dir():
                    click.echo(f"  {t.name}: path missing ({resolved})")
                    continue
                git_status = get_git_status(str(resolved))
                stored_sha = state.get_repo_index(t.name, profile)["sha"]
                if stored_sha and git_status and stored_sha == git_status.sha and not full:
                    click.echo(f"  {t.name}: up to date ({git_status.branch}, {git_status.sha[:7]})")
                else:
                    mode = "full" if full or not stored_sha else "incremental"
                    git_files = _git_tracked_files(str(resolved))
                    if git_files is not None:
                        supported = [f for f in git_files
                                     if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
                                     and Path(f).name not in SKIP_FILENAMES]
                        click.echo(f"  {t.name}: {mode} — {len(supported)} supported files in {resolved}")
                    else:
                        click.echo(f"  {t.name}: {mode} — not a git repo, cannot estimate")
        return

    from flowmap.embeddings import create_backend
    from flowmap.services.indexing import index_changed_content, run_index
    from flowmap.state import StateDB
    from flowmap.store import VectorStore

    try:
        backend = create_backend(
            backend=cfg.embedding.backend,
            model=cfg.embedding.model,
            ollama_url=cfg.embedding.ollama_url,
        )
    except (ConnectionError, ValueError, ImportError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    profile = cfg.embedding.profile_name
    click.echo(f"Embedding: {backend.model_name()} ({backend.dims()} dims) [profile: {profile}]")

    # Acquire process-level lock to prevent concurrent index corruption
    lock_path = cfg.data_path / ".flowmap.lock"
    lock_fd = _acquire_index_lock(lock_path)
    if lock_fd is None:
        click.echo("Error: another flowmap index is running. Wait or delete the lock file.", err=True)
        sys.exit(1)

    try:
        with StateDB(cfg.db_path) as state, VectorStore(cfg.lancedb_path, vector_dims=backend.dims()) as store:
            # Check model consistency *within this profile*. Switching models is
            # done by switching profiles (separate tables), so a mismatch here
            # means the profile's own model definition changed under it.
            stored_model = state.get_meta("embedding_model", profile)
            if stored_model and stored_model != backend.model_name():
                # If the profile has no data (e.g. after `reset --repo` emptied it),
                # there's nothing to be inconsistent with — allow the switch without
                # forcing --full.
                profile_empty = store.get_stats(profile=profile).get("total", 0) == 0
                if not full and not profile_empty:
                    click.echo(
                        f"Error: model for profile '{profile}' changed "
                        f"({stored_model} -> {backend.model_name()}). "
                        f"Run with --full to re-index.",
                        err=True,
                    )
                    sys.exit(1)

            results = run_index(
                store=store,
                state=state,
                backend=backend,
                targets=targets,
                full=full,
                on_message=lambda msg: click.echo(msg),
                profile=profile,
            )

            # Rebuild only when the corpus changed (the O(corpus) rebuild has no
            # partial update) — OR when the on-disk FTS index predates the current
            # schema, a one-time forced migration so stale profiles don't keep a
            # positionless index that silently returns [] for phrase queries.
            from flowmap.store import FTS_INDEX_VERSION
            fts_outdated = state.get_meta("fts_index_version", profile) != FTS_INDEX_VERSION
            if index_changed_content(results) or fts_outdated:
                if index_changed_content(results):
                    click.echo("Rebuilding search indexes...")
                else:
                    click.echo("Upgrading search index to current schema...")
                built_version = store.rebuild_fts_index(profile)
                store.rebuild_vector_index(profile)
                store.compact(profile)
                # Stamp what was actually built (not the constant): a "1" fallback
                # or None build leaves the profile outdated to re-try next run.
                if built_version is not None:
                    state.set_meta("fts_index_version", built_version, profile)
            click.echo("Done.")
    finally:
        _release_index_lock(lock_fd)


# ---------------------------------------------------------------------------
# flowmap search
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query")
@click.option("--repo", default=None, help="Filter by repo name")
@click.option("--limit", default=10, help="Number of results")
@click.option("--mode", type=click.Choice(["hybrid", "semantic", "keyword", "symbol"]), default="hybrid")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("--rerank", is_flag=True, default=False, help="Enable cross-encoder reranking (slower, higher quality)")
@click.option("--regex", "use_regex", is_flag=True, default=False, help="Treat keyword query as regex (default: literal match)")
@click.pass_context
def search(ctx, query, repo, limit, mode, fmt, rerank, use_regex):
    """Search across indexed repositories."""
    cfg = _load_cfg(ctx)

    from flowmap.embeddings import create_backend
    from flowmap.render import (
        render_hybrid_results,
        render_keyword_results,
        render_semantic_results,
        render_symbol_results,
    )
    from flowmap.store import StoreError, VectorStore

    # Embedding backend needed for hybrid/semantic modes
    backend = None
    if mode in ("hybrid", "semantic"):
        try:
            backend = create_backend(
                backend=cfg.embedding.backend,
                model=cfg.embedding.model,
                ollama_url=cfg.embedding.ollama_url,
            )
        except (ConnectionError, ValueError, ImportError) as e:
            if mode == "hybrid":
                click.echo(f"Warning: embedding unavailable ({e}), falling back to keyword-only.", err=True)
                mode = "keyword"
            else:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

    # --- Keyword only (ripgrep) — no vector store needed ---
    if mode == "keyword":
        from flowmap.search.ripgrep import rg_search

        repo_paths = cfg.repo_paths()
        if repo:
            repo_paths = {k: v for k, v in repo_paths.items() if k == repo}

        results = rg_search(query, repo_paths, limit=limit, regex=use_regex)
        if not results:
            if fmt == "json":
                click.echo(render_keyword_results([], query, fmt))
            else:
                click.echo("No results found.")
            return
        click.echo(render_keyword_results(results, query, fmt))
        return

    # Modes that need the vector store
    if backend:
        store_dims = backend.dims()
    else:
        store_dims = _get_stored_dims(cfg)
    try:
      with VectorStore(cfg.lancedb_path, vector_dims=store_dims) as store:

        # Warn loudly when the active profile has no index — otherwise empty
        # results look like "no matches" rather than "this model isn't indexed".
        active_profile = cfg.embedding.profile_name
        if fmt != "json" and active_profile not in store.list_profiles():
            click.echo(
                f"Warning: profile '{active_profile}' is not indexed. "
                f"Run: flowmap index",
                err=True,
            )
        # Warn on model drift: the profile was indexed with a different model than
        # config now points at. Dims still validate, but a same-dim swap silently
        # queries the wrong vectors. Only meaningful when we embed (backend exists).
        elif fmt != "json" and backend is not None:
            from flowmap.state import StateDB
            with StateDB(cfg.db_path) as _state:
                stored_model = _state.get_meta("embedding_model", active_profile)
            if stored_model and stored_model != backend.model_name():
                click.echo(
                    f"Warning: profile '{active_profile}' was indexed with "
                    f"{stored_model}, but config now uses {backend.model_name()}. "
                    f"Re-index or results will be wrong.",
                    err=True,
                )

        # --- Hybrid mode (default): 4-way fusion (ripgrep + BM25/FTS + vector + symbol) ---
        if mode == "hybrid":
            from flowmap.search.hybrid import hybrid_search

            repo_paths = cfg.repo_paths()
            if repo:
                repo_paths = {k: v for k, v in repo_paths.items() if k == repo}

            results = hybrid_search(
                query=query,
                repo_paths=repo_paths,
                embedding_backend=backend,
                store=store,
                limit=limit,
                repo_filter=repo,
                reranking_enabled=rerank or cfg.reranking.enabled,
                reranking_model=cfg.reranking.model,
                regex=use_regex,
                profile=cfg.embedding.profile_name,
            )

            if not results:
                if fmt == "json":
                    click.echo(render_hybrid_results([], query, fmt))
                else:
                    click.echo("No results found.")
                return
            click.echo(render_hybrid_results(results, query, fmt))

        # --- Semantic only ---
        elif mode == "semantic":
            query_vector = backend.embed_query(query)
            results = store.search_vector(query_vector, limit=limit, repo_filter=repo, profile=cfg.embedding.profile_name)

            if not results:
                if fmt == "json":
                    click.echo(render_semantic_results([], query, fmt))
                else:
                    click.echo("No results found.")
                return
            click.echo(render_semantic_results(results, query, fmt))

        # --- Symbol lookup ---
        elif mode == "symbol":
            results = store.search_symbol(query, repo_filter=repo, limit=limit, profile=cfg.embedding.profile_name)
            if not results:
                if fmt == "json":
                    click.echo(render_symbol_results([], query, fmt))
                else:
                    click.echo("No symbols found.")
                return
            click.echo(render_symbol_results(results, query, fmt))

    except StoreError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Try: flowmap index --full", err=True)
        sys.exit(1)



# ---------------------------------------------------------------------------
# flowmap map
# ---------------------------------------------------------------------------

@main.command("map")
@click.option("--repo", default=None, help="Show map for a specific repo")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def repo_map(ctx, repo, fmt):
    """Show structural overview of indexed repos."""
    cfg = _load_cfg(ctx)

    from flowmap.render import render_map
    from flowmap.services.map_builder import build_repo_map
    from flowmap.store import StoreError, VectorStore

    try:
        with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
            rows, truncated = store.get_repo_map(repo=repo, profile=cfg.embedding.profile_name)

            if not rows:
                if fmt == "json":
                    click.echo(render_map([], fmt))
                elif repo:
                    click.echo(f"No indexed data for repo '{repo}'. Run: flowmap index --repo {repo}")
                else:
                    click.echo("No indexed data. Run: flowmap index")
                return

            if truncated:
                click.echo("Warning: results truncated at 50,000 entries. Use --repo to filter.", err=True)

            output_repos = build_repo_map(rows)
            click.echo(render_map(output_repos, fmt))
    except StoreError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Try: flowmap index --full", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# flowmap symbols
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query", required=False, default=None)
@click.option("--repo", default=None, help="Filter by repo name")
@click.option("--type", "kind", default=None, type=click.Choice(["class", "function", "method", "property"]), help="Filter by symbol type")
@click.option("--limit", default=50, help="Max results")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def symbols(ctx, query, repo, kind, limit, fmt):
    """List symbols across indexed repos."""
    cfg = _load_cfg(ctx)

    from flowmap.render import render_symbols
    from flowmap.store import StoreError, VectorStore

    try:
        with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
            rows = store.get_symbols(query=query, repo=repo, kind=kind, limit=limit, profile=cfg.embedding.profile_name)

            if not rows:
                if fmt == "json":
                    click.echo(render_symbols([], query, fmt))
                else:
                    click.echo("No symbols found.")
                return

            click.echo(render_symbols(rows, query, fmt))
    except StoreError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Try: flowmap index --full", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# flowmap cat
# ---------------------------------------------------------------------------

@main.command()
@click.argument("file_path")
@click.option("--repo", default=None, help="Repo name (auto-detected if file path is inside a configured repo)")
@click.option("--lines", default=None, help="Line range, e.g. '10-50' or '42'")
@click.option("--symbol", default=None, help="Show the chunk containing this symbol name")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def cat(ctx, file_path, repo, lines, symbol, fmt):
    """Read file content from a configured repo.

    Show full source code for files found via search/symbols.
    Supports line ranges and symbol-based lookup.

    \b
    Examples:
      flowmap cat zed-dispatch-service/src/dispatch.service.ts
      flowmap cat src/auth.py --repo my-service --lines 25-70
      flowmap cat src/service.ts --repo my-service --symbol processTaxiDispatch
      flowmap cat zed-dispatch-service/src/workflow.ts --format json
    """
    cfg = _load_cfg(ctx)
    import json as json_mod

    from flowmap.services.file_resolver import resolve_file
    from flowmap.services.symbol_lookup import get_symbol_suggestions, resolve_symbol

    # Resolve the file to a repo
    try:
        resolved = resolve_file(file_path, cfg.repos, explicit_repo=repo)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    repo_cfg = resolved.repo_cfg
    repo_root = resolved.repo_root
    abs_file = resolved.abs_file
    rel = resolved.rel_file

    if not abs_file.exists():
        click.echo(f"File not found: {abs_file}", err=True)
        sys.exit(1)

    # If --symbol, find the chunk and show its line range
    if symbol:
        from flowmap.store import StoreError, VectorStore

        try:
            with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
                match = resolve_symbol(symbol, repo_cfg.name, rel, store, profile=cfg.embedding.profile_name)

                if not match:
                    suggestions = get_symbol_suggestions(repo_cfg.name, rel, store, profile=cfg.embedding.profile_name)
                    click.echo(f"Symbol '{symbol}' not found in {rel}.", err=True)
                    if suggestions:
                        click.echo(f"  Available symbols: {', '.join(suggestions)}", err=True)
                    else:
                        click.echo("  Try without --symbol to read the full file.", err=True)
                    sys.exit(1)

                lines = f"{match.result.start_line}-{match.result.end_line}"
                click.echo(f"# {match.result.symbol_name} at {match.result.repo}/{match.result.file}:{match.result.start_line}-{match.result.end_line}", err=True)
        except StoreError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Read the file
    try:
        content = abs_file.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        click.echo(f"Error reading file: {e}", err=True)
        sys.exit(1)

    file_lines = content.splitlines()

    # Apply line range
    if lines:
        try:
            if "-" in lines:
                parts = lines.split("-", 1)
                start = max(1, int(parts[0]))
                end = min(len(file_lines), int(parts[1]))
            else:
                start = max(1, int(lines))
                end = min(len(file_lines), start + 50)  # default: 50 lines from start
        except ValueError:
            click.echo(f"Invalid line range '{lines}'. Use format: 10-50 or 42", err=True)
            sys.exit(1)

        selected = file_lines[start - 1:end]

        if fmt == "json":
            output = {
                "repo": repo_cfg.name,
                "file": rel,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(selected),
            }
            click.echo(json_mod.dumps(output, indent=2))
        else:
            for i, line in enumerate(selected, start):
                click.echo(f"{i:>4}  {line}")
    else:
        if fmt == "json":
            output = {
                "repo": repo_cfg.name,
                "file": rel,
                "start_line": 1,
                "end_line": len(file_lines),
                "content": content,
            }
            click.echo(json_mod.dumps(output, indent=2))
        else:
            for i, line in enumerate(file_lines, 1):
                click.echo(f"{i:>4}  {line}")


# ---------------------------------------------------------------------------
# flowmap history
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query")
@click.option("--since", default="6 months ago", help="Time window (e.g. '3 months ago', '2025-01-01')")
@click.option("--repo", default=None, help="Filter by repo name")
@click.option("--limit", default=20, help="Max timeline entries")
@click.option("--symbol", default=None, help="Focus on a specific symbol name")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def history(ctx, query, since, repo, limit, symbol, fmt):
    """Show timeline of structural changes for a query.

    Scopes to relevant files via flowmap's index, then queries git history
    and compares versions with tree-sitter for AST-level diffs.

    \b
    Examples:
      flowmap history "validateToken"
      flowmap history "payment" --repo payment-service --since "3 months ago"
      flowmap history "OrderProcessor" --symbol OrderProcessor.process --limit 10
    """
    cfg = _load_cfg(ctx)

    from flowmap.history.timeline import build_timeline
    from flowmap.render import render_timeline
    from flowmap.store import StoreError, VectorStore

    # Create embedding backend for vector fallback (build_timeline uses it only if symbol search fails)
    from flowmap.embeddings import create_backend

    backend = None
    try:
        backend = create_backend(
            backend=cfg.embedding.backend,
            model=cfg.embedding.model,
            ollama_url=cfg.embedding.ollama_url,
        )
    except (ConnectionError, ValueError, ImportError) as e:
        click.echo(f"Note: embedding unavailable ({e}), history limited to symbol matches.", err=True)

    try:
        with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
            timeline = build_timeline(
                query=query,
                repo_paths=cfg.repo_paths(),
                store=store,
                since=since,
                limit=limit,
                repo_filter=repo,
                symbol_filter=symbol,
                embedding_backend=backend,
                profile=cfg.embedding.profile_name,
            )
    except StoreError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Try: flowmap index --full", err=True)
        sys.exit(1)

    if not timeline.entries:
        if fmt == "json":
            click.echo(render_timeline(timeline, fmt))
        else:
            click.echo(f"No history found for '{query}'.")
            if not timeline.scoped_files:
                click.echo("  No matching files in the index. Try a different query or run: flowmap index")
        return

    click.echo(render_timeline(timeline, fmt))


# ---------------------------------------------------------------------------
# flowmap status
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def status(ctx):
    """Show index status for all repos."""
    cfg = _load_cfg(ctx)
    if not cfg.repos:
        click.echo("No repos configured. Run: flowmap repos add /path/to/repo")
        return

    from flowmap.state import StateDB
    from flowmap.store import VectorStore

    profile = cfg.embedding.profile_name
    with StateDB(cfg.db_path) as state, VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
        stats = store.get_stats(profile=profile, known_repos=[r.name for r in cfg.repos])
        other = [p for p in store.list_profiles() if p != profile]
        header = f"Index [profile: {profile}]: {stats['total']} total chunks"
        if other:
            header += f"   (other profiles: {', '.join(sorted(other))})"
        click.echo(header + "\n")

        for repo in cfg.repos:
            path_exists = repo.resolved_path().is_dir()
            idx = state.get_repo_index(repo.name, profile)
            sha_val = idx["sha"]
            branch_val = idx["branch"]
            when = (idx["indexed_at"] or "")[:10]

            if not path_exists:
                marker = click.style("!", fg="red")
                click.echo(f"  {marker} {repo.name:<28} path missing: {repo.path}")
            elif sha_val:
                marker = click.style("✓", fg="green")
                chunks = stats.get("repos", {}).get(repo.name, 0)
                branch_str = f"{branch_val}, " if branch_val else ""
                click.echo(f"  {marker} {repo.name:<28} {chunks:>6} chunks  {when}  ({branch_str}{sha_val[:7]})")
            else:
                marker = click.style("✗", fg="yellow")
                click.echo(f"  {marker} {repo.name:<28} not indexed (profile: {profile})")


# ---------------------------------------------------------------------------
# flowmap reset
# ---------------------------------------------------------------------------

@main.command()
@click.option("--repo", default=None, help="Reset a specific repo")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--benchmarks", is_flag=True, help="Remove benchmark profiles only")
@click.confirmation_option(prompt="This will delete index data. Continue?")
@click.pass_context
def reset(ctx, repo, reset_all, benchmarks):
    """Delete index data."""
    cfg = _load_cfg(ctx)

    from flowmap.state import StateDB
    from flowmap.store import VectorStore

    with StateDB(cfg.db_path) as state, VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
        if benchmarks:
            # Only drop on-disk profiles that are NOT in config — true throwaway
            # benchmark tables. Never delete a configured profile or default.
            configured = set(cfg.embedding.profiles) | {"default"}
            for profile in store.list_profiles():
                if profile not in configured:
                    store.drop_table(profile)
                    state.clear_profile_staleness(profile)
                    click.echo(f"Dropped benchmark profile: {profile}")
        elif repo:
            # Purge the repo from every profile table + clear all its staleness.
            for prof in store.list_profiles():
                store.delete_by_repo(repo, profile=prof)
            state.delete_repo(repo)  # atomic: repos row + per-profile staleness
            click.echo(f"Reset: {repo}")
        elif reset_all:
            for profile in store.list_profiles():
                store.drop_table(profile)
            for r in state.list_repos():
                state.delete_repo(r["name"])
            # Belt-and-suspenders: clear any orphaned staleness for repos already
            # gone from the repos table (the exact pre-fix bug class).
            state.clear_all_staleness()
            click.echo("Reset: all index data deleted.")
        else:
            click.echo("Specify --repo, --all, or --benchmarks.", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# flowmap doctor
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def doctor(ctx):
    """Check system health: repos, index, embedding backend, dependencies."""
    cfg = _load_cfg(ctx)
    import shutil

    ok = True

    # 1. Check repos
    click.echo("Repos:")
    if not cfg.repos:
        click.echo("  No repos configured. Run: flowmap repos add /path/to/repo")
        ok = False
    else:
        for repo in cfg.repos:
            if repo.resolved_path().is_dir():
                click.echo(f"  {click.style('OK', fg='green')}  {repo.name} -> {repo.path}")
            else:
                click.echo(f"  {click.style('MISSING', fg='red')}  {repo.name} -> {repo.path}")
                ok = False

    # 2. Check embedding backend
    click.echo("\nEmbedding backend:")
    try:
        from flowmap.embeddings import create_backend
        backend = create_backend(
            backend=cfg.embedding.backend,
            model=cfg.embedding.model,
            ollama_url=cfg.embedding.ollama_url,
        )
        backend_dims = backend.dims()
        click.echo(f"  {click.style('OK', fg='green')}  {backend.model_name()} ({backend_dims} dims)")
    except (ConnectionError, ValueError, ImportError) as e:
        click.echo(f"  {click.style('ERROR', fg='red')}  {e}")
        ok = False
        backend_dims = None

    # 3. Check ripgrep
    click.echo("\nDependencies:")
    if shutil.which("rg"):
        click.echo(f"  {click.style('OK', fg='green')}  ripgrep (rg) installed")
    else:
        click.echo(f"  {click.style('MISSING', fg='yellow')}  ripgrep (rg) not installed — keyword search disabled")

    # 4. Check index state (for the active profile)
    profile = cfg.embedding.profile_name
    click.echo(f"\nIndex [profile: {profile}]:")
    from flowmap.state import StateDB
    from flowmap.store import VectorStore
    try:
        with StateDB(cfg.db_path) as state:
            stored_model = state.get_meta("embedding_model", profile)
            stored_dims = state.get_meta("embedding_dims", profile)
            if stored_model:
                click.echo(f"  Model: {stored_model} ({stored_dims or '?'} dims)")
                if stored_dims and backend_dims and int(stored_dims) != backend_dims:
                    click.echo(f"  {click.style('WARN', fg='yellow')}  Dimension mismatch: stored={stored_dims}, current={backend_dims}. Run: flowmap index --full")
                    ok = False
            else:
                click.echo(f"  {click.style('EMPTY', fg='yellow')}  No index data for profile '{profile}'. Run: flowmap index")
                ok = False

            # Check for pending markers
            for repo in cfg.repos:
                pending = state.get_meta(f"pending:{repo.name}", profile)
                if pending:
                    click.echo(f"  {click.style('WARN', fg='yellow')}  {repo.name}: interrupted index detected (will re-index on next run)")

        with VectorStore(cfg.lancedb_path, vector_dims=_get_stored_dims(cfg)) as store:
            stats = store.get_stats(profile=profile, known_repos=[r.name for r in cfg.repos])
            click.echo(f"  Total chunks: {stats['total']}")
            for rname, count in stats.get("repos", {}).items():
                click.echo(f"    {rname}: {count} chunks")
    except Exception as e:
        click.echo(f"  {click.style('ERROR', fg='red')}  {e}")
        ok = False

    # Summary
    click.echo()
    if ok:
        click.echo(click.style("All checks passed.", fg="green"))
    else:
        click.echo(click.style("Some checks failed. See above.", fg="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
