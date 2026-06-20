"""Embedding backends for FlowMap — Ollama (default) and sentence-transformers (optional)."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import requests

log = logging.getLogger(__name__)


def _err_detail(err: Exception | None) -> str:
    """HTTP status + body for an embedding failure, when present."""
    if err is None:
        return ""
    resp = getattr(err, "response", None)
    if resp is not None:
        return f" Last response: HTTP {resp.status_code} {resp.text[:200]}"
    return f" {type(err).__name__}: {err}"


# batch_size lowered 32 -> 8 to dodge the Ollama 0.30.10 runner crash on large
# embedding requests. Real fix was pinning 0.30.9; can be raised back to 32 once
# a healthy runner is guaranteed.
EMBED_BATCH_SIZE = 8
EMBED_TRANSIENT_RETRIES = 3

# ---------------------------------------------------------------------------
# Model-specific prefix requirements
# ---------------------------------------------------------------------------

# Each model has its own prefix format baked into training.
# Using wrong prefixes silently degrades retrieval quality.
KNOWN_PREFIXES: dict[str, dict[str, str]] = {
    # Qwen3-Embedding: uses Instruct/Query format
    "qwen3-embedding:0.6b": {
        "document": "",
        "query": "Instruct: Given a code search query, retrieve relevant code snippets\nQuery: ",
    },
    "qwen3-embedding": {
        "document": "",
        "query": "Instruct: Given a code search query, retrieve relevant code snippets\nQuery: ",
    },
    # Nomic embed text (general purpose, not code-specific)
    "nomic-embed-text": {
        "document": "search_document: ",
        "query": "search_query: ",
    },
    # CodeRankEmbed (sentence-transformers)
    "nomic-ai/CodeRankEmbed": {
        "document": "",
        "query": "Represent this query for searching relevant code: ",
    },
    # SFR-Embedding-Code (sentence-transformers)
    "Salesforce/SFR-Embedding-Code-400M_R": {
        "document": "",
        "query": "",
    },
    # Jina Code v2 (sentence-transformers)
    "jinaai/jina-embeddings-v2-base-code": {
        "document": "",
        "query": "",
    },
}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def dims(self) -> int: ...
    def model_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

_ollama_lock = threading.Lock()
_ollama_checked: dict[str, bool] = {}  # cache: (url, model) → verified


class OllamaBackend:
    """Ollama-based embedding backend via HTTP API."""

    def __init__(self, model: str = "qwen3-embedding:0.6b", url: str = "http://localhost:11434"):
        self._model = model
        self._url = url.rstrip("/")
        self._prefixes = KNOWN_PREFIXES.get(model, {"document": "", "query": ""})
        self._dims: int | None = None
        self._session = requests.Session()
        cache_key = f"{self._url}|{self._model}"
        with _ollama_lock:
            if cache_key not in _ollama_checked:
                self._check_available()
                _ollama_checked[cache_key] = True

    def _check_available(self):
        """Verify Ollama is running and model is pulled."""
        try:
            resp = self._session.get(f"{self._url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama not running at {self._url}. Start it with: ollama serve"
            )
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self._url}: {e}")

        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Check exact name or exact base name (before tag) match
        model_base = self._model.split(":")[0]
        if not any(
            m == self._model or m.split(":")[0] == model_base
            for m in models
        ):
            raise ValueError(
                f"Model '{self._model}' not found in Ollama. Run: ollama pull {self._model}"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = self._prefixes["document"]
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        return self._embed_batch(prefixed)

    def embed_query(self, text: str) -> list[float]:
        prefix = self._prefixes["query"]
        prefixed = f"{prefix}{text}" if prefix else text
        result = self._embed_batch([prefixed])
        return result[0]

    def dims(self) -> int:
        if self._dims is None:
            # Probe with a small input to detect dimensions
            result = self._embed_batch(["test"])
            self._dims = len(result[0])
        return self._dims

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def model_name(self) -> str:
        return f"ollama:{self._model}"

    def _embed_batch(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
        """Embed texts via Ollama /api/embed in small sub-batches, each recovered
        by _embed_recover so one bad input can't fail the whole run."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            embeddings = self._embed_recover(texts[i:i + batch_size])
            if self._dims is None and embeddings:
                self._dims = len(embeddings[0])
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _post_embed(self, batch: list[str], truncate: bool) -> list[list[float]]:
        """One /api/embed call, embeddings 1:1 with `batch`, or raises.

        truncate defaults False: truncate=True yields a prefix-only vector
        (lossy), so it's only the last-resort backstop in _embed_recover.
        """
        resp = self._session.post(
            f"{self._url}/api/embed",
            json={"model": self._model, "input": batch, "truncate": truncate},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(batch):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs"
            )
        return embeddings

    def _embed_recover(self, batch: list[str]) -> list[list[float]]:
        """Embed a batch with recovery: retry, then (HTTP error) bisect to
        isolate the bad input, then truncate a lone offender as a last resort."""
        import time

        transient_retries = EMBED_TRANSIENT_RETRIES
        last_err: Exception | None = None
        for attempt in range(transient_retries):
            try:
                return self._post_embed(batch, truncate=False)
            except requests.Timeout:
                raise TimeoutError(
                    f"Ollama embedding timed out after 120s (batch size: {len(batch)}). "
                    "Try reducing batch size or check Ollama resource usage."
                )
            except (requests.ConnectionError, requests.HTTPError) as e:
                last_err = e
                if attempt < transient_retries - 1:
                    time.sleep(min(2 ** attempt, 8))

        # A connection that survives every retry means the runner is unreachable,
        # not that one input is bad — fail fast rather than bisect into a dead server.
        if isinstance(last_err, requests.ConnectionError):
            raise ConnectionError(
                f"Ollama unreachable at {self._url} after {transient_retries} attempts "
                f"(batch size: {len(batch)}).{_err_detail(last_err)}"
            ) from last_err

        # HTTP error survived retries (e.g. over-context input): bisect to isolate it.
        if len(batch) > 1:
            mid = len(batch) // 2
            log.warning(
                "Embedding batch of %d failed (%s); bisecting to isolate the bad input.",
                len(batch), _err_detail(last_err).strip(),
            )
            return self._embed_recover(batch[:mid]) + self._embed_recover(batch[mid:])

        # Lone input still failing — truncate as a last resort (prefix-only vector).
        try:
            result = self._post_embed(batch, truncate=True)
            log.warning(
                "Embedded a single %d-char input only with truncation (prefix-only vector).",
                len(batch[0]),
            )
            return result
        except (requests.ConnectionError, requests.HTTPError, ValueError) as e:
            raise ConnectionError(
                f"Ollama embedding failed at {self._url} for a single input "
                f"({len(batch[0])} chars), even with truncation.{_err_detail(e)}"
            ) from e


# ---------------------------------------------------------------------------
# Sentence-transformers backend
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    """Optional backend. Requires: pip install flowmap[local-embeddings]"""

    def __init__(self, model: str = "nomic-ai/CodeRankEmbed"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install flowmap[local-embeddings]"
            )
        self._lock = threading.Lock()
        self._model_name = model
        self._prefixes = KNOWN_PREFIXES.get(model, {"document": "", "query": ""})
        log.info("Loading model: %s (first run downloads weights)", model)
        with self._lock:
            self._model = SentenceTransformer(model, trust_remote_code=True)
        self._dims_val = self._model.get_sentence_embedding_dimension()
        if self._dims_val is None:
            # Fallback: probe with a test input for models that don't report dims
            probe = self._model.encode(["test"])
            self._dims_val = len(probe[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = self._prefixes["document"]
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        with self._lock:
            return self._model.encode(prefixed, batch_size=32, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = self._prefixes["query"]
        prefixed = f"{prefix}{text}" if prefix else text
        with self._lock:
            return self._model.encode([prefixed])[0].tolist()

    def dims(self) -> int:
        return self._dims_val

    def model_name(self) -> str:
        return f"st:{self._model_name}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(backend: str, model: str, ollama_url: str = "http://localhost:11434") -> EmbeddingBackend:
    """Create an embedding backend from config values."""
    if backend == "ollama":
        return OllamaBackend(model=model, url=ollama_url)
    elif backend == "sentence-transformers":
        return SentenceTransformerBackend(model=model)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}. Use 'ollama' or 'sentence-transformers'.")
