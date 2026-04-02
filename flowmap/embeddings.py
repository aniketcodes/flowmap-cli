import random
import sys
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from flowmap.config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMS

# https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types
# Index = "documents" (code chunks); ask = NL query optimized for code retrieval
_TASK_INDEX = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "CODE_RETRIEVAL_QUERY"

_client = None

_MAX_RETRIES = 10
_BASE_DELAY = 2.0
_MAX_DELAY = 90.0


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Export it: export GOOGLE_API_KEY='your_key'"
            )
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _is_rate_limited(err: ClientError) -> bool:
    code = getattr(err, "code", None)
    if code == 429:
        return True
    msg = str(err).lower()
    return "429" in msg or "resource_exhausted" in msg


def _embed_config(task_type: str) -> types.EmbedContentConfig:
    return types.EmbedContentConfig(
        output_dimensionality=EMBEDDING_DIMS,
        task_type=task_type,
    )


def _embed_batch_once(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=_embed_config(_TASK_INDEX),
    )
    return [e.values for e in response.embeddings]


def get_embeddings_batch(texts: list[str]) -> tuple[list[list[float]], bool]:
    """Returns (embeddings, had_any_429). Retries on 429 and splits batch if needed."""
    if not texts:
        return [], False
    return _embed_batch_with_retry(texts)


def _embed_batch_with_retry(texts: list[str]) -> tuple[list[list[float]], bool]:
    last_err: Exception | None = None
    saw_429 = False
    for attempt in range(_MAX_RETRIES):
        try:
            return _embed_batch_once(texts), saw_429
        except ClientError as e:
            last_err = e
            if not _is_rate_limited(e):
                raise
            saw_429 = True
            delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
            delay += random.uniform(0, 0.75)
            print(
                f"flowmap: Gemini rate limit (429); sleeping {delay:.1f}s "
                f"(retry {attempt + 1}/{_MAX_RETRIES}, batch size {len(texts)})...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    if last_err is not None:
        if len(texts) > 1:
            print(
                "flowmap: splitting embedding batch in half after repeated 429s...",
                file=sys.stderr,
                flush=True,
            )
            mid = len(texts) // 2
            left, l429 = _embed_batch_with_retry(texts[:mid])
            right, r429 = _embed_batch_with_retry(texts[mid:])
            return left + right, saw_429 or l429 or r429
        raise last_err
    raise RuntimeError("embedding batch failed")


def get_embedding(text: str) -> list[float]:
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            client = _get_client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=_embed_config(_TASK_QUERY),
            )
            return response.embeddings[0].values
        except ClientError as e:
            last_err = e
            if not _is_rate_limited(e):
                raise
            delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
            delay += random.uniform(0, 0.75)
            time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("embedding failed")
