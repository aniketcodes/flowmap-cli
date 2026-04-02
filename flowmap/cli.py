import argparse
import os
import sys
import time

from tqdm import tqdm

from flowmap.config import (
    BATCH_SIZE,
    EMBED_COOLDOWN_AFTER_429,
    EMBED_INTER_BATCH_DELAY,
)
from flowmap.indexer import index_repo
from flowmap.embeddings import get_embedding, get_embeddings_batch
from flowmap.store import ensure_collection, upsert_chunks, search


def _resolve_repo_path(raw: str) -> str:
    """Resolve a path for indexing; fixes common typo: Users/... vs /Users/..."""
    raw = raw.strip()
    expanded = os.path.expanduser(raw)
    candidate = os.path.abspath(expanded)
    if os.path.isdir(candidate):
        return candidate
    # macOS: absolute home path typed without leading slash
    if raw.startswith("Users/") and os.path.isdir("/" + raw):
        return "/" + raw
    return candidate


def cmd_index(args):
    repo_path = _resolve_repo_path(args.path)
    if not os.path.isdir(repo_path):
        print(f"Error: not a directory: {args.path}")
        print(f"Resolved path: {repo_path}")
        print("Hint: use an absolute path, e.g. /Users/you/project (note the leading /).")
        sys.exit(1)

    print(f"Indexing: {repo_path}")
    chunks = index_repo(repo_path)
    if not chunks:
        print("No supported files found (no matching extensions or empty files).")
        print("See flowmap/config.py → SUPPORTED_EXTENSIONS")
        return

    print(f"Found {len(chunks)} chunks")

    ensure_collection()

    texts = [c["text"] for c in chunks]
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Generating embeddings: {len(texts)} chunks in {n_batches} batches "
        f"(batch size {BATCH_SIZE}).",
        flush=True,
    )
    print(
        "The progress bar stays at 0% until the first batch returns — that call "
        "often takes 30–120s (cold start). 429 rate limits add retries and longer waits.",
        flush=True,
    )
    embeddings = []
    for i in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Batches",
        total=n_batches,
        unit="batch",
    ):
        batch = texts[i : i + BATCH_SIZE]
        batch_embs, had_429 = get_embeddings_batch(batch)
        embeddings.extend(batch_embs)
        if had_429 and EMBED_COOLDOWN_AFTER_429 > 0:
            print(
                f"Pausing {EMBED_COOLDOWN_AFTER_429:.0f}s after rate limits before the next batch...",
                flush=True,
            )
            time.sleep(EMBED_COOLDOWN_AFTER_429)
        if i + BATCH_SIZE < len(texts) and EMBED_INTER_BATCH_DELAY > 0:
            time.sleep(EMBED_INTER_BATCH_DELAY)

    print("Storing in Qdrant...")
    upsert_chunks(chunks, embeddings)
    print(f"Done. Indexed {len(chunks)} chunks from {repo_path}")


def cmd_ask(args):
    query = args.query

    query_vector = get_embedding(query)
    results = search(query_vector, limit=args.limit, repo_filter=args.repo)

    if not results:
        print("No results found.")
        return

    for i, hit in enumerate(results, 1):
        p = hit.payload
        preview = p["text"].replace("\n", " ")[:200]
        print(f"[{i}] {p['repo']}/{p['file']}  (score: {hit.score:.4f})")
        print(f"    {preview}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="flowmap", description="Cross-repo code intelligence"
    )
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index a repository")
    idx.add_argument("path", help="Path to the repository")

    ask = sub.add_parser("ask", help="Search the codebase")
    ask.add_argument("query", help="Search query")
    ask.add_argument("--limit", type=int, default=5, help="Number of results")
    ask.add_argument("--repo", default=None, help="Filter by repo name")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
