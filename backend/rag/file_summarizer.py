"""
backend/rag/file_summarizer.py
-------------------------------
Automatic file intelligence generation.

Public API
──────────
  generate_file_intelligence(text, model_name)  — main entry point (full pipeline)
  summarize_file(text, model_name)              — summary only
  generate_tags(text, model_name)               — tags only
  generate_suggested_questions(text, model_name)— questions only

All functions are safe to call individually or together via generate_file_intelligence().
Each has its own safe default on LLM failure.
"""

import logging
from backend.llm import call_llm, safe_json_parse
from backend.prompts import FILE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# ── Token budget ──────────────────────────────────────────────────────────────
# Send enough context for a meaningful summary but not more.
INTELLIGENCE_CHAR_LIMIT = 5000
DEFAULT_SUMMARY = "No summary available."
DEFAULT_QUESTIONS = [
    "De quoi parle ce document ?",
    "Quels sont les points clés de ce document ?",
]


def _clean_text_preview(text: str, max_chars: int = 420) -> str:
    preview = " ".join((text or "").split())
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars].rsplit(" ", 1)[0] + "..."


def _safe_list(value, default: list[str], max_items: int = 8, max_chars: int = 80) -> list[str]:
    if not isinstance(value, list):
        return default
    cleaned = [str(item).strip()[:max_chars] for item in value if str(item).strip()]
    return cleaned[:max_items] or default


def _deterministic_file_intelligence(text: str) -> dict:
    """
    Last-resort metadata when the LLM is unavailable.
    Keep it factual: expose an extracted preview instead of inventing a summary.
    """
    preview = _clean_text_preview(text)
    summary = f"Aperçu extrait automatiquement : {preview}" if preview else DEFAULT_SUMMARY
    return {
        "summary": summary,
        "tags": ["document"],
        "key_topics": [],
        "suggested_questions": DEFAULT_QUESTIONS,
    }


# ── Individual generators ─────────────────────────────────────────────────────

def summarize_file(text: str, model_name: str = None) -> str:
    """
    Generate a short (≤ 5 sentence) factual summary of the document text.
    Returns a plain string. Falls back to a safe default on any error.
    """
    truncated = text[:INTELLIGENCE_CHAR_LIMIT]
    prompt = (
        f"Document Text:\n{truncated}\n\n"
        "Task: Write a factual summary of this document in 3 to 5 short sentences. "
        "Base yourself ONLY on the text above. "
        "Return ONLY the summary as a plain string — no JSON, no bullet points."
    )
    system = (
        "You are a document summarizer. Your output must be a plain text summary, "
        "3 to 5 sentences max. Do not use outside knowledge. Do not invent facts."
    )
    try:
        raw = call_llm(prompt=prompt, system=system, timeout=30, model_name=model_name)
        summary = raw.strip().strip('"').strip()
        return summary if len(summary) > 20 else DEFAULT_SUMMARY
    except Exception as e:
        logger.warning("summarize_file failed: %s", e)
        return DEFAULT_SUMMARY


def generate_tags(text: str, model_name: str = None) -> list[str]:
    """
    Generate 3–8 short tag labels for the document.
    Returns a list of strings. Falls back to ["document"] on failure.
    """
    truncated = text[:INTELLIGENCE_CHAR_LIMIT]
    prompt = (
        f"Document Text:\n{truncated}\n\n"
        "Task: Generate 3 to 8 short tag labels for this document. "
        "Examples: 'finance', 'technical', 'report', 'dataset', 'requirements', 'PFA', 'business'. "
        "Return ONLY a valid JSON array of strings. Example: [\"finance\", \"report\", \"2024\"]"
    )
    system = (
        "You are a document tagger. Return ONLY a JSON array of short strings. "
        "No markdown. No explanation. 3 to 8 items max."
    )
    try:
        raw = call_llm(prompt=prompt, system=system, timeout=20, model_name=model_name)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, list) and parsed:
            # Sanitise: keep only non-empty strings ≤ 30 chars
            return [str(t)[:30] for t in parsed if t][:8]
    except Exception as e:
        logger.warning("generate_tags failed: %s", e)
    return ["document"]


def generate_suggested_questions(text: str, model_name: str = None) -> list[str]:
    """
    Generate 5 user-friendly questions that can be answered from the document.
    Returns a list of strings. Falls back to a generic question on failure.
    """
    truncated = text[:INTELLIGENCE_CHAR_LIMIT]
    prompt = (
        f"Document Text:\n{truncated}\n\n"
        "Task: Generate exactly 5 useful, specific questions a user could ask about this document. "
        "The questions must be answerable from the document content. "
        "Write in the same language as the document. "
        "Return ONLY a valid JSON array of 5 strings. "
        "Example: [\"Quels sont les objectifs ?\", \"Quelles sont les parties prenantes ?\"]"
    )
    system = (
        "You are a document intelligence assistant. "
        "Return ONLY a JSON array of exactly 5 question strings. "
        "No markdown. No numbering. No extra text."
    )
    try:
        raw = call_llm(prompt=prompt, system=system, timeout=30, model_name=model_name)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, list) and parsed:
            questions = [str(q) for q in parsed if q][:5]
            if questions:
                return questions
    except Exception as e:
        logger.warning("generate_suggested_questions failed: %s", e)
    return [DEFAULT_QUESTIONS[0]]


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_file_intelligence(text: str, model_name: str = None) -> dict:
    """
    Run the full file intelligence pipeline in a single LLM call.

    Tries the efficient single-call approach (FILE_SUMMARY_PROMPT) first.
    Falls back to three individual calls if parsing fails.

    Returns
    ───────
    dict with keys: summary, tags, key_topics, suggested_questions
    All keys are guaranteed to be present.
    """
    truncated = text[:INTELLIGENCE_CHAR_LIMIT]
    user_prompt = (
        f"Document Text:\n{truncated}\n\n"
        "Generate the JSON intelligence metadata for this document."
    )

    parsed = None
    try:
        raw_response = call_llm(
            prompt     = user_prompt,
            system     = FILE_SUMMARY_PROMPT,
            timeout    = 60,
            model_name = model_name,
        )
        parsed = safe_json_parse(raw_response)
    except Exception as e:
        logger.warning("generate_file_intelligence: single-call LLM failed: %s", e)

    # Validate the single-call result
    if (
        parsed
        and isinstance(parsed, dict)
        and "summary" in parsed
        and len(parsed.get("summary", "")) > 10
    ):
        result = {
            "summary"            : str(parsed.get("summary", DEFAULT_SUMMARY)),
            "tags"               : _safe_list(parsed.get("tags"), ["document"], max_items=8, max_chars=30),
            "key_topics"         : _safe_list(parsed.get("key_topics"), [], max_items=8, max_chars=80),
            "suggested_questions": _safe_list(parsed.get("suggested_questions"), DEFAULT_QUESTIONS, max_items=5, max_chars=180),
        }
        logger.info(
            "generate_file_intelligence: single-call success (tags=%d, questions=%d)",
            len(result["tags"]), len(result["suggested_questions"]),
        )
        return result

    # ── Fallback: three individual calls ──────────────────────────────────────
    logger.warning(
        "generate_file_intelligence: single-call parse failed, falling back to individual calls"
    )
    fallback = _deterministic_file_intelligence(text)

    summary = summarize_file(text, model_name)
    if summary != DEFAULT_SUMMARY:
        fallback["summary"] = summary

    fallback["tags"] = generate_tags(text, model_name)
    if parsed and isinstance(parsed, dict):
        fallback["key_topics"] = _safe_list(parsed.get("key_topics"), [], max_items=8, max_chars=80)
    fallback["suggested_questions"] = generate_suggested_questions(text, model_name)

    return fallback
