"""
rag/pinecone_client.py — Pinecone vector database client.
Handles index initialization, upsert, and similarity search.
"""
from typing import List, Dict, Any, Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

log = structlog.get_logger()

# Lazy-loaded to avoid import errors if pinecone not configured
_pinecone_index = None


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            _pinecone_index = pc.Index(settings.pinecone_index_name)
            log.info("pinecone.index_connected", index=settings.pinecone_index_name)
        except Exception as e:
            log.error("pinecone.connection_failed", error=str(e))
            raise
    return _pinecone_index


def create_index_if_not_exists(dimension: int = 1536):
    """
    Create Pinecone index if it doesn't exist.
    Call this once during ingestion setup.
    dimension=1536 for text-embedding-ada-002
    dimension=768  for sentence-transformers
    """
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment,
            ),
        )
        log.info(
            "pinecone.index_created",
            name=settings.pinecone_index_name,
            dimension=dimension,
        )
    else:
        log.info("pinecone.index_exists", name=settings.pinecone_index_name)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def upsert_vectors(
    vectors: List[Dict[str, Any]],
    namespace: str = "",
    batch_size: int = 100,
) -> int:
    """
    Upsert vectors into Pinecone in batches.

    vectors: [{"id": str, "values": List[float], "metadata": dict}, ...]
    Returns total number of vectors upserted.
    """
    index = _get_index()
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)
        log.debug("pinecone.upsert_batch", batch_num=i // batch_size + 1, count=len(batch))
    log.info("pinecone.upsert_complete", total=total, namespace=namespace)
    return total


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def query_vectors(
    embedding: List[float],
    namespace: str = "",
    top_k: int = 5,
    filter: Optional[Dict] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for nearest neighbors.
    Returns list of matches with id, score, metadata.
    """
    index = _get_index()
    kwargs: Dict[str, Any] = {
        "vector": embedding,
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": True,
    }
    if filter:
        kwargs["filter"] = filter

    result = index.query(**kwargs)

    matches = [
        {
            "id": m["id"],
            "score": m["score"],
            "metadata": m.get("metadata", {}),
        }
        for m in result.get("matches", [])
        if m["score"] >= min_score
    ]
    return matches


def get_index_stats() -> Dict[str, Any]:
    """Return index statistics (total vectors, namespaces, etc.)."""
    index = _get_index()
    return index.describe_index_stats()
