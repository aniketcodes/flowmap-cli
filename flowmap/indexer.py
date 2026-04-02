import os
from pathlib import Path

from flowmap.config import SUPPORTED_EXTENSIONS, SKIP_DIRS, CHUNK_SIZE, CHUNK_OVERLAP

# Huge generated files — skip to avoid massive token use and slow embeds
SKIP_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Gemfile.lock",
}


def _is_supported_file(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_repo(repo_path: str) -> list[dict]:
    """Walk a repo, filter files, chunk text, return chunks with metadata."""
    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)
    chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            if filename in SKIP_FILENAMES:
                continue
            filepath = os.path.join(root, filename)
            if not _is_supported_file(filepath):
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, IOError):
                continue

            if not content.strip():
                continue

            rel_path = os.path.relpath(filepath, repo_path)
            file_chunks = _chunk_text(content)

            for i, chunk_text in enumerate(file_chunks):
                chunks.append(
                    {
                        "repo": repo_name,
                        "file": rel_path,
                        "file_name": filename,
                        "extension": Path(filename).suffix.lower(),
                        "chunk_index": i,
                        "text": chunk_text,
                    }
                )

    return chunks
