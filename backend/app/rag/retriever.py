"""
rag/retriever.py — Multi-namespace retrieval from Pinecone.

Strategy:
  For symptom queries  → weight causes_symptoms > overview > diagnosis
  For lab queries      → weight diagnosis > key_terms > treatment
  Results are merged and re-ranked by score.
"""
from typing import List, Dict, Any, Optional
import structlog

from app.config import settings
from app.rag import pinecone_client, embedder

log = structlog.get_logger()

# Namespace weights for different query types
QUERY_TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "symptom_analysis": {
        "causes_symptoms": 0.50,
        "overview":        0.25,
        "diagnosis":       0.15,
        "description":     0.10,
    },
    "lab_interpretation": {
        "diagnosis":       0.45,
        "key_terms":       0.25,
        "treatment":       0.20,
        "causes_symptoms": 0.10,
    },
    "general": {
        "overview":        0.35,
        "causes_symptoms": 0.30,
        "diagnosis":       0.20,
        "treatment":       0.15,
    },
}


class RetrievedChunk:
    def __init__(self, chunk_id: str, score: float, metadata: Dict[str, Any], namespace: str):
        self.chunk_id = chunk_id
        self.score = score
        self.metadata = metadata
        self.namespace = namespace

    @property
    def text(self) -> str:
        return self.metadata.get("text", "")

    @property
    def disease_name(self) -> str:
        return self.metadata.get("disease_name", "Unknown")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "unknown")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")


def retrieve(
    query_text: str,
    query_type: str = "symptom_analysis",
    top_k_per_namespace: int | None = None,
    total_top_k: int = 8,
    min_score: float | None = None,
    disease_filter: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Main retrieval function.

    1. Embed the query
    2. Query the default namespace (all vectors are in the default namespace)
    3. Filter by minimum score and return top-k
    """
    top_k_per_namespace = top_k_per_namespace or settings.rag_top_k
    min_score = min_score or settings.rag_min_score

    # Step 1: Embed query
    query_embedding = embedder.embed_single(query_text)

    pinecone_filter = {}
    if disease_filter:
        pinecone_filter = {"diseases_mentioned": {"$in": disease_filter}}

    # Step 2: Query default namespace (all data lives here)
    all_chunks: List[RetrievedChunk] = []
    try:
        matches = pinecone_client.query_vectors(
            embedding=query_embedding,
            namespace="",   # default namespace contains all 23k vectors
            top_k=max(total_top_k * 2, top_k_per_namespace * 4),
            filter=pinecone_filter if pinecone_filter else None,
            min_score=0.0,
        )

        for match in matches:
            chunk = RetrievedChunk(
                chunk_id=match["id"],
                score=match["score"],
                metadata=match["metadata"],
                namespace="default",
            )
            all_chunks.append(chunk)

    except Exception as e:
        log.error("retriever.query_failed", error=str(e))

    # Step 3: Filter by minimum score and sort
    filtered = [c for c in all_chunks if c.score >= min_score * 0.5]
    filtered.sort(key=lambda c: c.score, reverse=True)

    result = filtered[:total_top_k]

    log.info(
        "retriever.complete",
        query_type=query_type,
        total_candidates=len(all_chunks),
        returned=len(result),
        top_score=result[0].score if result else 0.0,
    )

    return result


def retrieve_for_symptoms(symptoms: List[str], context: Dict[str, Any] = {}) -> List[RetrievedChunk]:
    """
    High-level helper: build a semantic query from symptoms and retrieve.
    """
    symptom_str = ", ".join(symptoms)
    age = context.get("age", "")
    gender = context.get("gender", "")
    duration = context.get("duration_days", "")

    query = (
        f"Patient presents with: {symptom_str}. "
        f"{'Age: ' + str(age) + '.' if age else ''} "
        f"{'Gender: ' + str(gender) + '.' if gender else ''} "
        f"{'Duration: ' + str(duration) + ' days.' if duration else ''} "
        f"What disease conditions match these symptoms? Causes and diagnostic criteria."
    )

    return retrieve(query_text=query, query_type="symptom_analysis")


def retrieve_for_lab_values(lab_summary: str) -> List[RetrievedChunk]:
    """High-level helper: retrieve context for lab value interpretation."""
    query = (
        f"Laboratory test results interpretation: {lab_summary}. "
        f"What do these values indicate? Diagnosis and clinical significance."
    )
    return retrieve(query_text=query, query_type="lab_interpretation")
