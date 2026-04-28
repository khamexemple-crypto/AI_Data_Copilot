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
        return summary if len(summary) > 20 else "No summary available."
    except Exception as e:
        logger.warning("summarize_file failed: %s", e)
        return "No summary available."


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
    return ["De quoi parle ce document ?"]


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

    raw_response = call_llm(
        prompt     = user_prompt,
        system     = FILE_SUMMARY_PROMPT,
        timeout    = 60,
        model_name = model_name,
    )
    parsed = safe_json_parse(raw_response)

    # Validate the single-call result
    if (
        parsed
        and isinstance(parsed, dict)
        and "summary" in parsed
        and len(parsed.get("summary", "")) > 10
    ):
        result = {
            "summary"            : str(parsed.get("summary", "No summary available.")),
            "tags"               : parsed.get("tags", ["document"]) if isinstance(parsed.get("tags"), list) else ["document"],
            "key_topics"         : parsed.get("key_topics", []) if isinstance(parsed.get("key_topics"), list) else [],
            "suggested_questions": parsed.get("suggested_questions", ["De quoi parle ce document ?"]) if isinstance(parsed.get("suggested_questions"), list) else ["De quoi parle ce document ?"],
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
    return {
        "summary"            : summarize_file(text, model_name),
        "tags"               : generate_tags(text, model_name),
        "key_topics"         : parsed.get("key_topics", []) if parsed and isinstance(parsed, dict) else [],
        "suggested_questions": generate_suggested_questions(text, model_name),
    }
