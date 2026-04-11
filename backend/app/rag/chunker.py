"""
rag/chunker.py — Section-aware chunking for the Gale Encyclopedia of Medicine.

The Gale Encyclopedia has a consistent structure per disease entry:
  [Disease Name]
  Definition
  Description
  Causes and symptoms
    Causes / Symptoms
  Diagnosis
  Treatment
  Prevention
  KEY TERMS
  Resources

We chunk by section so each chunk is semantically coherent.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional
import structlog

log = structlog.get_logger()

# ── Section header patterns (order matters — more specific first) ──────────────
SECTION_PATTERNS = [
    ("key_terms",       re.compile(r"^KEY\s+TERMS\s*$", re.MULTILINE | re.IGNORECASE)),
    ("resources",       re.compile(r"^Resources?\s*$", re.MULTILINE | re.IGNORECASE)),
    ("prevention",      re.compile(r"^Prevention\s*$", re.MULTILINE | re.IGNORECASE)),
    ("treatment",       re.compile(r"^Treatment\s*$", re.MULTILINE | re.IGNORECASE)),
    ("diagnosis",       re.compile(r"^Diagnosis\s*$", re.MULTILINE | re.IGNORECASE)),
    ("causes_symptoms", re.compile(r"^Causes?\s+and\s+symptoms?\s*$", re.MULTILINE | re.IGNORECASE)),
    ("symptoms",        re.compile(r"^Symptoms?\s*$", re.MULTILINE | re.IGNORECASE)),
    ("causes",          re.compile(r"^Causes?\s*$", re.MULTILINE | re.IGNORECASE)),
    ("description",     re.compile(r"^Description\s*$", re.MULTILINE | re.IGNORECASE)),
    ("definition",      re.compile(r"^Definition\s*$", re.MULTILINE | re.IGNORECASE)),
]

# Entry boundaries — new disease entry starts when we see a standalone capitalized title
ENTRY_BOUNDARY = re.compile(
    r"^([A-Z][A-Za-z\s\-',\(\)]{2,60})\s*\n(?=\n?(?:Definition|Description|see also))",
    re.MULTILINE,
)

# Minimum chunk character length — shorter chunks are noise
MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 3000  # ~750 tokens, well within ada-002 limit


@dataclass
class MedicalChunk:
    """A single semantic chunk from the encyclopedia."""
    chunk_id: str
    disease_name: str
    section: str          # overview|definition|description|causes_symptoms|
                          # diagnosis|treatment|prevention|key_terms
    text: str             # full chunk content including disease name header
    page_start: int
    page_end: int
    char_count: int = field(init=False)

    # Extracted metadata for Pinecone filtering
    symptoms_mentioned: List[str] = field(default_factory=list)
    diseases_mentioned: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_pinecone_metadata(self) -> dict:
        return {
            "disease_name": self.disease_name,
            "section": self.section,
            "text": self.text[:40000],  # Pinecone metadata 40KB limit
            "page_number": self.page_start,
            "symptoms_mentioned": self.symptoms_mentioned[:20],
            "diseases_mentioned": self.diseases_mentioned[:10],
            "source": "Gale Encyclopedia of Medicine 3rd Edition",
            "char_count": self.char_count,
        }


def chunk_encyclopedia_text(
    full_text: str,
    page_map: Optional[List[tuple]] = None,  # [(page_num, char_offset), ...]
) -> List[MedicalChunk]:
    """
    Main chunking function. Takes raw extracted text from the PDF
    and returns a list of MedicalChunk objects.
    """
    chunks: List[MedicalChunk] = []
    disease_entries = _split_into_disease_entries(full_text)

    log.info("chunker.entries_found", count=len(disease_entries))

    for disease_name, entry_text, entry_start_char in disease_entries:
        entry_chunks = _chunk_disease_entry(
            disease_name=disease_name,
            entry_text=entry_text,
            entry_start_char=entry_start_char,
            page_map=page_map,
        )
        chunks.extend(entry_chunks)

    log.info("chunker.total_chunks", count=len(chunks))
    return chunks


def _split_into_disease_entries(text: str) -> List[tuple]:
    """
    Split the full encyclopedia text into individual disease entries.
    Returns: [(disease_name, entry_text, start_char_pos), ...]
    """
    entries = []

    # Find all entry boundaries
    matches = list(ENTRY_BOUNDARY.finditer(text))

    for i, match in enumerate(matches):
        disease_name = match.group(1).strip()

        # Skip obvious non-disease headers (volume headers, TOC, etc.)
        if _is_non_disease_header(disease_name):
            continue

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entry_text = text[start:end].strip()

        if len(entry_text) < MIN_CHUNK_CHARS:
            continue

        entries.append((disease_name, entry_text, start))

    return entries


def _chunk_disease_entry(
    disease_name: str,
    entry_text: str,
    entry_start_char: int,
    page_map: Optional[List[tuple]],
) -> List[MedicalChunk]:
    """Split a single disease entry into section-level chunks."""
    chunks = []

    # Find all sections within this entry
    section_positions = []
    for section_name, pattern in SECTION_PATTERNS:
        for m in pattern.finditer(entry_text):
            section_positions.append((m.start(), section_name, m.end()))

    # Sort by position
    section_positions.sort(key=lambda x: x[0])

    if not section_positions:
        # No recognizable sections — treat entire entry as "overview"
        chunk = _make_chunk(
            disease_name=disease_name,
            section="overview",
            text=entry_text[:MAX_CHUNK_CHARS],
            char_start=entry_start_char,
            page_map=page_map,
            index=0,
        )
        if chunk:
            chunks.append(chunk)
        return chunks

    # Create overview chunk: disease name + text before first section
    text_before_first_section = entry_text[: section_positions[0][0]].strip()
    if len(text_before_first_section) > MIN_CHUNK_CHARS:
        # Combine disease name + preamble as overview
        overview_text = f"{disease_name}\n\n{text_before_first_section}"
        chunk = _make_chunk(
            disease_name=disease_name,
            section="overview",
            text=overview_text[:MAX_CHUNK_CHARS],
            char_start=entry_start_char,
            page_map=page_map,
            index=0,
        )
        if chunk:
            chunks.append(chunk)

    # Create one chunk per section
    for idx, (sec_start, sec_name, sec_header_end) in enumerate(section_positions):
        # Section content: from end of header to start of next section (or end of entry)
        next_sec_start = (
            section_positions[idx + 1][0]
            if idx + 1 < len(section_positions)
            else len(entry_text)
        )
        section_content = entry_text[sec_header_end:next_sec_start].strip()

        if len(section_content) < MIN_CHUNK_CHARS:
            continue

        # Skip Resources section — not useful for RAG
        if sec_name == "resources":
            continue

        # Prepend disease name to every chunk for context
        chunk_text = f"{disease_name} — {sec_name.replace('_', ' ').title()}\n\n{section_content}"

        # If chunk is too long, split further on paragraph boundaries
        sub_chunks = _split_if_too_long(chunk_text, disease_name, sec_name)

        for sub_idx, sub_text in enumerate(sub_chunks):
            chunk = _make_chunk(
                disease_name=disease_name,
                section=sec_name,
                text=sub_text,
                char_start=entry_start_char + sec_start,
                page_map=page_map,
                index=idx * 10 + sub_idx,
            )
            if chunk:
                chunks.append(chunk)

    return chunks


def _make_chunk(
    disease_name: str,
    section: str,
    text: str,
    char_start: int,
    page_map: Optional[List[tuple]],
    index: int,
) -> Optional[MedicalChunk]:
    """Construct a MedicalChunk with metadata extraction."""
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return None

    page = _char_to_page(char_start, page_map)
    safe_disease = re.sub(r"[^\w]", "_", disease_name.lower())[:40]
    chunk_id = f"gale_{safe_disease}_{section}_{page}_{index}"

    symptoms = _extract_symptoms(text)
    diseases = _extract_disease_names(text)

    return MedicalChunk(
        chunk_id=chunk_id,
        disease_name=disease_name,
        section=section,
        text=text,
        page_start=page,
        page_end=page,
        symptoms_mentioned=symptoms,
        diseases_mentioned=diseases,
    )


def _split_if_too_long(text: str, disease_name: str, section: str) -> List[str]:
    """Split oversized chunks on paragraph boundaries with overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    sub_chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                sub_chunks.append(current.strip())
            # Overlap: include last paragraph of previous chunk
            overlap = current.split("\n\n")[-1] if current else ""
            current = (overlap + "\n\n" + para).strip() if overlap else para

    if current:
        sub_chunks.append(current.strip())

    return sub_chunks or [text[:MAX_CHUNK_CHARS]]


# ── Helper utilities ────────────────────────────────────────────────────────────

COMMON_SYMPTOMS = [
    "fatigue", "nausea", "vomiting", "fever", "pain", "headache",
    "dizziness", "cough", "shortness of breath", "chest pain",
    "abdominal pain", "diarrhea", "constipation", "weakness",
    "weight loss", "weight gain", "sweating", "chills", "loss of appetite",
    "frequent urination", "blurred vision", "rash", "swelling", "bleeding",
    "insomnia", "anxiety", "depression", "confusion", "numbness",
    "tingling", "palpitations", "jaundice", "anemia", "hypertension",
]

SYMPTOM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in COMMON_SYMPTOMS) + r")\b",
    re.IGNORECASE,
)


def _extract_symptoms(text: str) -> List[str]:
    found = set(m.group(1).lower() for m in SYMPTOM_PATTERN.finditer(text))
    return sorted(found)


def _extract_disease_names(text: str) -> List[str]:
    """Extract capitalized multi-word phrases as potential disease names."""
    pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Za-z]+){1,4})\b")
    found = set()
    for m in pattern.finditer(text):
        candidate = m.group(1)
        if len(candidate) > 5 and candidate not in {"Gale", "Encyclopedia", "Medicine"}:
            found.add(candidate)
    return sorted(found)[:10]


def _char_to_page(char_offset: int, page_map: Optional[List[tuple]]) -> int:
    """Convert character offset to approximate page number."""
    if not page_map:
        return 0
    page = 1
    for page_num, page_char_start in page_map:
        if char_offset >= page_char_start:
            page = page_num
        else:
            break
    return page


def _is_non_disease_header(name: str) -> bool:
    """Filter out table of contents, volume headers, etc."""
    skip = {
        "GALE ENCYCLOPEDIA", "List of Entries", "Volume", "Index",
        "Introduction", "Contributors", "Advisory Board", "Preface",
    }
    return any(s.lower() in name.lower() for s in skip) or len(name) < 3
