"""Shared test fixtures and helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

DIMS = 32


def hash_vector(text: str) -> list[float]:
    """Deterministic vector from text hash. For test roundtrips, not real embeddings."""
    h = hashlib.sha256(text.encode()).digest()
    return [float(b) / 255.0 for b in h[:DIMS]]


class MockBackend:
    """Deterministic hash-based embedding backend for testing."""

    def embed_documents(self, texts):
        return [hash_vector(t) for t in texts]

    def embed_query(self, text):
        return hash_vector(text)

    def dims(self):
        return DIMS

    def model_name(self):
        return "test:mock"


@dataclass
class FakeCommit:
    sha: str
    author: str
    date: str
    message: str


@dataclass
class FakeHistory:
    file: str
    commits: list = field(default_factory=list)
