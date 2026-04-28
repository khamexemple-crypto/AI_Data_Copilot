"""
backend/rag/source_validator.py
--------------------------------
Post-generation validation layer.

Responsibilities
────────────────
  1. detect_insufficient_context()  — decide if chunks are too weak before calling LLM
  2. compute_confidence()           — multi-factor score (rerank, diversity, length)
  3. detect_contradictions()        — flag conflicting filenames in used sources
  4. build_source_citations()       — build the canonical sources[] list with excerpts
  5. validate_sources()             — public orchestrator called by rag_agent.py

All functions are pure / side-effect-free except validate_sources which mutates
and returns the answer dict.
"""

import logging

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
INSUFFICIENT_TOP_SCORE   = 0.15   # top rerank_score below this → refuse to answer
INSUFFICIENT_MIN_CHUNKS  = 1      # need at least this many chunks
EXCERPT_MAX_CHARS        = 300    # max chars per excerpt snippet
WEAK_CONFIDENCE_CUTOFF   = 0.40   # below this we add a limitations warning


# ── 1. Insufficient context detection ────────────────────────────────────────

def detect_insufficient_context(retrieved_chunks: list) -> tuple[bool, str]:
    """
    Return (is_insufficient: bool, reason: str).

    Called BEFORE the LLM to short-circuit when context is clearly useless.
    """
    if not retrieved_chunks:
        return True, "No chunks retrieved from the document store."

    if len(retrieved_chunks) < INSUFFICIENT_MIN_CHUNKS:
        return True, f"Too few chunks retrieved ({len(retrieved_chunks)})."

    top_score = retrieved_chunks[0].get("rerank_score", 0.0)
    if top_score < INSUFFICIENT_TOP_SCORE:
        return True, (
            f"Top rerank score ({top_score:.3f}) is below the minimum threshold "
            f"({INSUFFICIENT_TOP_SCORE}). Context is too weak to answer reliably."
        )

    return False, ""


# ── 2. Confidence scoring ─────────────────────────────────────────────────────

def compute_confidence(retrieved_chunks: list, llm_grounded: bool) -> float:
    """
    Multi-factor confidence score in [0.0, 1.0].

    Factors
    ───────
      rerank_score   — quality of the best chunk          (40 %)
      chunk_count    — more chunks = more evidence        (20 %)
      source_diversity — multiple files = less bias       (20 %)
      context_length — enough text to actually answer     (20 %)

    Returns 0.0 immediately if LLM said grounded=false or no chunks.
    """
    if not llm_grounded or not retrieved_chunks:
        return 0.0

    # ── Factor 1: best rerank score (0–1 normalised)
    top_score   = max((c.get("rerank_score", 0.0) for c in retrieved_chunks), default=0.0)
    score_factor = min(top_score / 0.8, 1.0)   # saturates at 0.8

    # ── Factor 2: chunk count (saturates at 5)
    count_factor = min(len(retrieved_chunks) / 5, 1.0)

    # ── Factor 3: source diversity (number of unique files, saturates at 3)
    unique_files     = len({c.get("filename", "") for c in retrieved_chunks})
    diversity_factor = min(unique_files / 3, 1.0)

    # ── Factor 4: total context length (saturates at 1500 chars)
    total_text   = sum(len(c.get("text", "")) for c in retrieved_chunks)
    length_factor = min(total_text / 1500, 1.0)

    raw = (
        score_factor    * 0.40 +
        count_factor    * 0.20 +
        diversity_factor * 0.20 +
        length_factor   * 0.20
    )

    confidence = round(raw, 2)
    logger.debug(
        "Confidence factors — score=%.2f count=%.2f diversity=%.2f length=%.2f → %.2f",
        score_factor, count_factor, diversity_factor, length_factor, confidence,
    )
    return confidence


# ── 3. Contradiction detection ────────────────────────────────────────────────

def detect_contradictions(used_chunks: list) -> list[str]:
    """
    Very lightweight contradiction signal: if two *different* files are cited
    and they have conflicting numeric mentions, flag it.

    For now we use a heuristic: if >1 unique filename is in used_chunks return
    an advisory note so the LLM answer can mention it.
    Returns a list of warning strings (empty = no contradictions detected).
    """
    filenames = [c.get("filename", "") for c in used_chunks]
    unique    = set(filenames)
    if len(unique) > 1:
        return [
            f"Sources from multiple files detected ({', '.join(sorted(unique))}). "
            "Verify consistency — different documents may contain conflicting information."
        ]
    return []


# ── 4. Build citations list ───────────────────────────────────────────────────

def _extract_excerpt(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Return the first *max_chars* characters, trimmed to the last full word."""
    if len(text) <= max_chars:
        return text.strip()
    trimmed = text[:max_chars]
    last_space = trimmed.rfind(" ")
    return (trimmed[:last_space] if last_space > 0 else trimmed).strip() + "…"


def build_source_citations(retrieved_chunks: list, used_filenames: list) -> list[dict]:
    """
    Build the canonical sources[] list.

    Priority:
      1. If the LLM provided used_filenames, filter to only those chunks.
      2. Otherwise include all retrieved chunks (fallback).

    Each citation:
        { file_id, filename, chunk_id, excerpt }
    """
    if used_filenames:
        used_set = set(used_filenames)
        pool     = [c for c in retrieved_chunks if c.get("filename", "") in used_set]
    else:
        pool = list(retrieved_chunks)

    # De-duplicate by chunk_id (keep highest rerank_score)
    seen: dict[str, dict] = {}
    for chunk in pool:
        cid = str(chunk.get("chunk_id", ""))
        if cid not in seen or chunk.get("rerank_score", 0.0) > seen[cid].get("rerank_score", 0.0):
            seen[cid] = chunk

    citations = []
    for chunk in seen.values():
        citations.append({
            "file_id"  : str(chunk.get("file_id", "")),
            "filename" : str(chunk.get("filename", "")),
            "chunk_id" : str(chunk.get("chunk_id", "")),
            "excerpt"  : _extract_excerpt(chunk.get("text", "")),
            "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
            "rank"     : chunk.get("rank", 99),
        })

    # Sort citations by rank so most relevant appears first
    citations.sort(key=lambda x: x["rank"])
    return citations


# ── 5. Public orchestrator ────────────────────────────────────────────────────

def validate_sources(raw_answer_dict: dict, retrieved_chunks: list) -> dict:
    """
    Post-process the LLM answer dict:

      • Compute multi-factor confidence.
      • Build canonical sources[] with excerpts.
      • Detect contradictions and append to limitations.
      • Override answer + zero confidence if grounded=false.
      • Add weak-context advisory when confidence is low.

    Returns the mutated answer dict.
    """
    grounded        = raw_answer_dict.get("grounded", False)
    used_filenames  = raw_answer_dict.get("used_sources", [])

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = compute_confidence(retrieved_chunks, grounded)
    raw_answer_dict["confidence"] = confidence

    # ── Not grounded / zero confidence path ───────────────────────────────────
    if not grounded or confidence == 0.0:
        raw_answer_dict["answer"] = (
            "Je n'ai pas trouvé assez d'informations dans les fichiers "
            "stockés pour répondre avec certitude."
        )
        raw_answer_dict["sources"]    = []
        raw_answer_dict["grounded"]   = False
        raw_answer_dict["confidence"] = 0.0
        lims = raw_answer_dict.setdefault("limitations", [])
        if "No sufficient context found." not in lims:
            lims.append("No sufficient context found.")
        logger.info("validate_sources → not grounded (confidence=0.0)")
        return raw_answer_dict

    # ── Build citations ───────────────────────────────────────────────────────
    citations = build_source_citations(retrieved_chunks, used_filenames)
    raw_answer_dict["sources"] = citations

    # ── Contradiction check ───────────────────────────────────────────────────
    used_pool = [c for c in retrieved_chunks
                 if not used_filenames or c.get("filename", "") in set(used_filenames)]
    contradictions = detect_contradictions(used_pool)
    if contradictions:
        raw_answer_dict.setdefault("limitations", []).extend(contradictions)
        logger.info("validate_sources → contradictions detected: %s", contradictions)

    # ── Low-confidence advisory ───────────────────────────────────────────────
    if confidence < WEAK_CONFIDENCE_CUTOFF:
        raw_answer_dict.setdefault("limitations", []).append(
            f"Confidence is low ({confidence:.2f}). "
            "The retrieved context may be incomplete — treat this answer with caution."
        )
        logger.warning("validate_sources → low confidence=%.2f", confidence)

    logger.info(
        "validate_sources → grounded=True, confidence=%.2f, sources=%d",
        confidence, len(citations),
    )
    return raw_answer_dict
