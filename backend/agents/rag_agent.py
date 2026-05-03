"""
backend/agents/rag_agent.py
----------------------------
Strictly-grounded RAG generation agent.

Design contract
───────────────
  • Answers ONLY from the provided Context block — no outside knowledge.
  • Returns a structured JSON dict matching the canonical schema.
  • Delegates all post-generation validation to source_validator.validate_sources().
  • Short-circuits BEFORE calling the LLM when context is clearly insufficient,
    using source_validator.detect_insufficient_context().
"""

import logging
from backend.llm import call_llm, safe_json_parse
from backend.rag.source_validator import (
    validate_sources,
    detect_insufficient_context,
    build_source_citations,
)
from backend.prompts import build_rag_prompt

logger = logging.getLogger(__name__)

# ── Canonical "no context" response ──────────────────────────────────────────
_NO_CONTEXT_RESPONSE = {
    "answer"    : "Je n'ai pas trouvé assez d'informations dans les fichiers stockés pour répondre avec certitude.",
    "sources"   : [],
    "confidence": 0.0,
    "limitations": ["No sufficient context found."],
    "grounded"  : False,
}


def _format_context_block(retrieved_chunks: list) -> str:
    """
    Build the numbered context block injected into the user prompt.

    Each chunk is presented as:
        [1] Source: filename.pdf | Chunk abc123
        <text>

    Numbering lets the LLM reference chunks by index in its answer.
    """
    lines = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        header = (
            f"[{idx}] Source: {chunk.get('filename', 'unknown')} "
            f"| Chunk {chunk.get('chunk_id', '?')} "
            f"| ReRank={chunk.get('rerank_score', 0.0):.3f}"
        )
        lines.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n---\n\n".join(lines)


def run_rag_agent(
    question: str,
    retrieved_chunks: list,
    model_name: str = None,
) -> dict:
    """
    Generate a grounded answer from *retrieved_chunks*.

    Pipeline
    ────────
      1. detect_insufficient_context() — short-circuit if chunks are too weak.
      2. Build numbered context block + structured system prompt.
      3. Call LLM, parse JSON.
      4. validate_sources()  — compute confidence, build citations, check contradictions.

    Returns
    ───────
    dict:
        answer       str
        sources      list[{file_id, filename, chunk_id, excerpt, rerank_score, rank}]
        confidence   float  (0.0 – 1.0)
        limitations  list[str]
        grounded     bool
    """
    # ── Guard 1: empty list ───────────────────────────────────────────────────
    if not retrieved_chunks:
        logger.warning("run_rag_agent: no chunks provided → returning no-context response")
        return dict(_NO_CONTEXT_RESPONSE, limitations=["No context retrieved."])

    # ── Guard 2: context quality check (before spending LLM tokens) ───────────
    insufficient, reason = detect_insufficient_context(retrieved_chunks)
    if insufficient:
        logger.warning("run_rag_agent: insufficient context — %s", reason)
        return dict(
            _NO_CONTEXT_RESPONSE,
            limitations=[reason, "No sufficient context found."],
        )

    # ── Build prompt ──────────────────────────────────────────────────────────
    context_block = _format_context_block(retrieved_chunks)
    system_prompt, user_prompt = build_rag_prompt(question, context_block)

    logger.info(
        "run_rag_agent: sending %d chunks to LLM (top_rerank=%.3f)",
        len(retrieved_chunks),
        retrieved_chunks[0].get("rerank_score", 0.0),
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        raw_response = call_llm(
            prompt     = user_prompt,
            system     = system_prompt,
            timeout    = 90,
            model_name = model_name,
        )
        parsed = safe_json_parse(raw_response)
    except Exception as e:
        logger.warning("run_rag_agent: LLM call failed after retrieval: %s", e)
        return {
            "answer": (
                "Des passages pertinents ont été retrouvés, mais le LLM est indisponible "
                "ou a échoué pendant la génération de la réponse."
            ),
            "sources": build_source_citations(retrieved_chunks, []),
            "confidence": 0.0,
            "limitations": [f"LLM generation failed: {e}"],
            "grounded": False,
            "llm_error": True,
        }

    # ── Parse failure fallback ─────────────────────────────────────────────────
    if not parsed:
        logger.error("run_rag_agent: LLM response could not be parsed as JSON")
        return {
            "answer": (
                "Des passages pertinents ont été retrouvés, mais la réponse du LLM "
                "n'était pas exploitable."
            ),
            "sources": build_source_citations(retrieved_chunks, []),
            "confidence": 0.0,
            "limitations": ["LLM returned an unparseable response. Please retry."],
            "grounded": False,
            "llm_error": True,
        }

    # ── Normalise LLM output keys ─────────────────────────────────────────────
    # Ensure required keys exist even if the LLM skipped them
    parsed.setdefault("answer",      "")
    parsed.setdefault("grounded",    False)
    parsed.setdefault("used_sources", [])
    parsed.setdefault("limitations", [])

    # Safety: if LLM claims grounded but answer is empty, override
    if parsed["grounded"] and not parsed["answer"].strip():
        logger.warning("run_rag_agent: LLM grounded=true but answer is empty — overriding")
        parsed["grounded"] = False
        parsed["limitations"].append("LLM returned grounded=true with an empty answer.")

    # ── Post-generation validation & citation building ────────────────────────
    validated = validate_sources(parsed, retrieved_chunks)
    return validated
