"""Shared test fixtures and helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

DIMS = 32


def hash_vector(text: str, dims: int = DIMS) -> list[float]:
    """Deterministic vector of arbitrary length from a text hash. For test
    roundtrips, not real embeddings. Chains counter-salted sha256 blocks so it
    supports real model sizes (e.g. 1024, 2560), not just <=32."""
    out: list[float] = []
    counter = 0
    while len(out) < dims:
        block = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        out.extend(float(b) / 255.0 for b in block)
        counter += 1
    return out[:dims]


class MockBackend:
    """Deterministic hash-based embedding backend for testing.

    Parametrizable by model name and dims so tests can exercise model-drift
    detection (same dims, different model) and dimension-mismatch validation
    (different dims) without a real embedding server.
    """

    def __init__(self, model_name: str = "test:mock", dims: int = DIMS):
        self._model_name = model_name
        self._dims = dims

    def embed_documents(self, texts):
        return [hash_vector(t, self._dims) for t in texts]

    def embed_query(self, text):
        return hash_vector(text, self._dims)

    def dims(self):
        return self._dims

    def model_name(self):
        return self._model_name


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
