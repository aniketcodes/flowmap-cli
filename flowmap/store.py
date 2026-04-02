import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from flowmap.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_DIMS

_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection():
    client = _get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMS,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    client = _get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "repo": chunk["repo"],
                "file": chunk["file"],
                "file_name": chunk["file_name"],
                "extension": chunk["extension"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )


def search(
    query_vector: list[float],
    limit: int = 5,
    repo_filter: str | None = None,
) -> list:
    client = _get_client()
    query_filter = None
    if repo_filter:
        query_filter = Filter(
            must=[FieldCondition(key="repo", match=MatchValue(value=repo_filter))]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )
    return response.points
