"""
rag/ingest_pipeline.py — One-time ingestion of the Gale Encyclopedia into Pinecone.

Run this ONCE (or whenever updating the knowledge base):
  python -m app.rag.ingest_pipeline --pdf /path/to/gale.pdf

Takes ~2-4 hours for the full 4,505-page PDF.
Progress is saved to allow resuming if interrupted.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import structlog

log = structlog.get_logger()

PROGRESS_FILE = "ingestion_progress.json"
CHUNK_CACHE_FILE = "chunks_cache.json"


def run_ingestion(pdf_path: str, resume: bool = True, dry_run: bool = False):
    """
    Full ingestion pipeline:
    1. Extract text from PDF
    2. Chunk into sections
    3. Embed with OpenAI
    4. Upsert to Pinecone
    """
    from app.rag.pinecone_client import create_index_if_not_exists, upsert_vectors, get_index_stats
    from app.rag.chunker import chunk_encyclopedia_text
    from app.rag.embedder import embed_texts, count_tokens

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        log.error("ingestion.pdf_not_found", path=str(pdf_path))
        sys.exit(1)

    log.info("ingestion.start", pdf=str(pdf_path))

    # ── Step 1: Extract text ────────────────────────────────────────────────────
    log.info("ingestion.step1_extract_text")
    full_text, page_map = _extract_pdf_text(str(pdf_path))
    log.info("ingestion.text_extracted", chars=len(full_text), pages=len(page_map))

    # ── Step 2: Chunk ───────────────────────────────────────────────────────────
    log.info("ingestion.step2_chunk")

    if resume and Path(CHUNK_CACHE_FILE).exists():
        log.info("ingestion.loading_chunk_cache")
        with open(CHUNK_CACHE_FILE) as f:
            chunk_dicts = json.load(f)
        from app.rag.chunker import MedicalChunk
        chunks = [_dict_to_chunk(d) for d in chunk_dicts]
    else:
        chunks = chunk_encyclopedia_text(full_text, page_map)
        # Save chunk cache
        with open(CHUNK_CACHE_FILE, "w") as f:
            json.dump([_chunk_to_dict(c) for c in chunks], f)

    log.info("ingestion.chunks_ready", total=len(chunks))

    if dry_run:
        log.info("ingestion.dry_run_complete", chunks=len(chunks))
        _print_sample_chunks(chunks, n=5)
        return

    # ── Step 3: Create Pinecone index ───────────────────────────────────────────
    log.info("ingestion.step3_create_index")
    create_index_if_not_exists(dimension=1536)
    time.sleep(10)  # Wait for index to be ready

    # ── Step 4: Load progress ───────────────────────────────────────────────────
    progress = _load_progress()
    start_idx = progress.get("last_upserted_idx", 0)
    log.info("ingestion.resuming_from", idx=start_idx)

    # ── Step 5: Embed + upsert in batches ──────────────────────────────────────
    log.info("ingestion.step4_embed_upsert")
    EMBED_BATCH = 50  # embed 50 chunks at a time

    for i in range(start_idx, len(chunks), EMBED_BATCH):
        batch_chunks = chunks[i : i + EMBED_BATCH]
        texts = [c.text for c in batch_chunks]

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            log.error("ingestion.embed_failed", batch=i, error=str(e))
            _save_progress({"last_upserted_idx": i})
            raise

        # Group by namespace (section)
        namespace_batches: dict = {}
        for chunk, embedding in zip(batch_chunks, embeddings):
            ns = chunk.section
            if ns not in namespace_batches:
                namespace_batches[ns] = []
            namespace_batches[ns].append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": chunk.to_pinecone_metadata(),
            })

        # Upsert each namespace
        for namespace, vectors in namespace_batches.items():
            upsert_vectors(vectors, namespace=namespace)

        _save_progress({"last_upserted_idx": i + len(batch_chunks)})

        pct = (i + len(batch_chunks)) / len(chunks) * 100
        log.info(
            "ingestion.progress",
            processed=i + len(batch_chunks),
            total=len(chunks),
            pct=f"{pct:.1f}%",
        )

        # Respect OpenAI rate limits
        time.sleep(0.5)

    # ── Step 6: Verify ──────────────────────────────────────────────────────────
    log.info("ingestion.step5_verify")
    stats = get_index_stats()
    log.info("ingestion.complete", index_stats=stats)

    # Clean up progress file on success
    for f in [PROGRESS_FILE]:
        if Path(f).exists():
            os.remove(f)

    log.info("🎉 Ingestion complete!", total_chunks=len(chunks))


def _extract_pdf_text(pdf_path: str):
    """Extract text from PDF using pdfplumber with page tracking."""
    try:
        import pdfplumber
    except ImportError:
        log.error("ingestion.pdfplumber_not_installed")
        raise

    full_text = ""
    page_map = []  # [(page_num, char_offset_start)]

    log.info("ingestion.extracting_pdf", path=pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        log.info("ingestion.total_pages", pages=total_pages)

        for i, page in enumerate(pdf.pages):
            char_start = len(full_text)
            page_map.append((i + 1, char_start))

            try:
                text = page.extract_text() or ""
                full_text += text + "\n\n"
            except Exception as e:
                log.warning("ingestion.page_extract_failed", page=i + 1, error=str(e))
                full_text += "\n\n"

            if i % 100 == 0:
                log.info("ingestion.extraction_progress", page=i + 1, total=total_pages)

    return full_text, page_map


def _load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def _save_progress(data: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def _chunk_to_dict(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "disease_name": chunk.disease_name,
        "section": chunk.section,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "symptoms_mentioned": chunk.symptoms_mentioned,
        "diseases_mentioned": chunk.diseases_mentioned,
    }


def _dict_to_chunk(d: dict):
    from app.rag.chunker import MedicalChunk
    return MedicalChunk(
        chunk_id=d["chunk_id"],
        disease_name=d["disease_name"],
        section=d["section"],
        text=d["text"],
        page_start=d["page_start"],
        page_end=d["page_end"],
        symptoms_mentioned=d.get("symptoms_mentioned", []),
        diseases_mentioned=d.get("diseases_mentioned", []),
    )


def _print_sample_chunks(chunks, n=5):
    print(f"\n{'='*60}")
    print(f"SAMPLE CHUNKS (first {n}):")
    print(f"{'='*60}")
    for chunk in chunks[:n]:
        print(f"\nID: {chunk.chunk_id}")
        print(f"Disease: {chunk.disease_name}")
        print(f"Section: {chunk.section}")
        print(f"Chars: {len(chunk.text)}")
        print(f"Preview: {chunk.text[:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Gale Encyclopedia into Pinecone")
    parser.add_argument("--pdf", required=True, help="Path to Gale Encyclopedia PDF")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore cache)")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only, don't embed/upsert")
    args = parser.parse_args()

    run_ingestion(
        pdf_path=args.pdf,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
