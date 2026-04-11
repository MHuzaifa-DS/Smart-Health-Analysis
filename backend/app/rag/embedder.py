"""
rag/embedder.py — Text embedding using OpenAI text-embedding-ada-002.
Handles batching, rate limiting, and caching during ingestion.
"""
from typing import List, Union
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

log = structlog.get_logger()

_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """
    Embed a list of strings. Returns list of float vectors.
    Automatically handles OpenAI batching limits (max 2048 texts per call).
    """
    model = model or settings.embedding_model
    client = _get_client()

    # OpenAI limit: 2048 inputs per request
    all_embeddings: List[List[float]] = []
    batch_size = 200

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text to ~8000 tokens max (ada-002 context window)
        batch = [t[:30000] for t in batch]

        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        log.debug(
            "embedder.batch_complete",
            batch=i // batch_size + 1,
            count=len(batch),
        )

    return all_embeddings


def embed_single(text: str, model: str | None = None) -> List[float]:
    """Embed a single string. Optimized for query-time use."""
    return embed_texts([text], model)[0]


def count_tokens(text: str) -> int:
    """Approximate token count for a text string."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return len(text) // 4
