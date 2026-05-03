"""
backend/rag/document_comparator.py
-----------------------------------
Multi-file comparison engine.
"""

import logging
from backend.llm import call_llm, safe_json_parse
from backend.rag.retriever import retrieve_context
from backend.rag.source_validator import validate_sources

logger = logging.getLogger(__name__)

_COMPARATOR_PROMPT = """You are a precise, objective document comparator.

Your task: Compare the provided chunks from different documents to answer the user's question.

RULES:
1. ONLY use the information provided in the Context blocks. Do NOT invent facts.
2. Clearly identify common points, differences, and contradictions between the documents.
3. Reference chunks by their [number]. Do NOT merge citations ambiguously.
4. If a contradiction is uncertain, state it as a "possible contradiction".
5. If the documents cannot be compared or don't answer the question, set "comparable" to false.

Return ONLY a valid JSON object:
{
  "common_points": ["Common point 1 citing [1]", "Common point 2 citing [2]"],
  "differences": ["Difference 1 citing [1] and [3]"],
  "contradictions": ["Contradiction: Doc A says X [1], Doc B says Y [3]"],
  "missing_information": ["Doc A does not mention Z"],
  "used_sources": ["filename1.pdf", "filename2.docx"],
  "comparable": true,
  "confidence": 0.8
}

If not comparable:
{
  "common_points": [],
  "differences": [],
  "contradictions": [],
  "missing_information": ["Documents lack sufficient information for comparison."],
  "used_sources": [],
  "comparable": false,
  "confidence": 0.0
}
"""

def _format_grouped_context(retrieved_chunks: list) -> str:
    # Group by file identity, not just filename. Two uploaded documents can have
    # the same original filename while still being distinct sources.
    groups = {}
    for c in retrieved_chunks:
        file_id = c.get("file_id", "unknown")
        fname = c.get("filename", "unknown")
        groups.setdefault((file_id, fname), []).append(c)

    lines = []
    idx = 1
    # Assign global chunk numbering so citations work the same way as standard RAG
    # We update the original list dicts with 'rank' if we want, but numbering here is enough.
    for (file_id, fname), chunks in groups.items():
        lines.append(f"=== Document: {fname} | file_id: {file_id} ===")
        for chunk in chunks:
            # We save this assigned index into the chunk dict temporarily to match later if needed,
            # but standard source validator works by filename. 
            chunk["prompt_idx"] = idx
            header = f"[{idx}] Chunk {chunk.get('chunk_id', '?')} | ReRank={chunk.get('rerank_score', 0.0):.3f}"
            lines.append(f"{header}\n{chunk.get('text', '').strip()}\n")
            idx += 1
    
    return "\n".join(lines)


def _excerpt(text: str, max_chars: int = 220) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _fallback_comparison(chunks: list, reason: str) -> dict:
    """
    Deterministic fallback when the LLM fails to return parseable JSON.
    It keeps the endpoint useful instead of returning no_context after retrieval succeeded.
    """
    groups = {}
    for chunk in chunks:
        key = chunk.get("file_id") or chunk.get("filename", "unknown")
        groups.setdefault(key, {
            "filename": chunk.get("filename", "unknown"),
            "chunks": [],
        })["chunks"].append(chunk)

    if len(groups) < 2:
        return {
            "common_points": [],
            "differences": [],
            "contradictions": [],
            "missing_information": [reason, "Retrieved context came from fewer than two distinct files."],
            "used_sources": [],
            "comparable": False,
            "confidence": 0.0,
            "fallback_used": True,
        }

    normalized_by_file = {
        file_key: {chunk.get("text", "").strip() for chunk in data["chunks"] if chunk.get("text")}
        for file_key, data in groups.items()
    }
    common_texts = set.intersection(*normalized_by_file.values()) if normalized_by_file else set()
    used_sources = [file_key for file_key in groups]

    common_points = []
    if common_texts:
        common_points.append(
            f"Selected documents share matching retrieved context: {_excerpt(next(iter(common_texts)))}"
        )
    else:
        filenames = [data["filename"] for data in groups.values()]
        common_points.append(
            f"Retrieved context was found in all selected documents ({', '.join(filenames)}), "
            "but no identical chunks were detected."
        )

    differences = []
    file_items = list(groups.items())
    for file_key, data in file_items[:3]:
        other_texts = set().union(*[
            texts for other_key, texts in normalized_by_file.items()
            if other_key != file_key
        ])
        unique = normalized_by_file[file_key] - other_texts
        if unique:
            differences.append(
                f"{data['filename']} has unique retrieved context: {_excerpt(next(iter(unique)))}"
            )

    return {
        "common_points": common_points,
        "differences": differences,
        "contradictions": [],
        "missing_information": [reason, "Fallback comparison used because the LLM response was not valid JSON."],
        "used_sources": used_sources,
        "comparable": True,
        "confidence": 0.45,
        "fallback_used": True,
    }


def compare_files(question: str, file_ids: list[str], model_name: str = None) -> dict:
    """
    Retrieve and compare chunks from specific files.
    """
    if not file_ids or len(file_ids) < 2:
        return {
            "status": "partial_success",
            "question": question,
            "common_points": [],
            "differences": [],
            "contradictions": [],
            "missing_information": ["At least two files are required for comparison."],
            "sources": [],
            "confidence": 0.0
        }

    # 1. Retrieve across selected files
    retrieval = retrieve_context(question, file_ids=file_ids)
    chunks = retrieval.get("chunks", [])

    if not chunks:
        return {
            "status": "no_context",
            "question": question,
            "common_points": [],
            "differences": [],
            "contradictions": [],
            "missing_information": ["No relevant context found in the selected files."],
            "sources": [],
            "confidence": 0.0
        }

    # 2. Build grouped prompt
    context_block = _format_grouped_context(chunks)
    user_prompt = f"Question: {question}\n\nContext:\n\n{context_block}"

    # 3. Call LLM
    try:
        raw = call_llm(prompt=user_prompt, system=_COMPARATOR_PROMPT, timeout=90, model_name=model_name)
        parsed = safe_json_parse(raw)
    except Exception as e:
        logger.warning("compare_files: LLM call failed, using fallback — %s", e)
        parsed = _fallback_comparison(chunks, f"LLM comparison error: {e}")

    if not parsed:
        parsed = _fallback_comparison(chunks, "LLM parse error.")

    # 4. Use source validator logic (mocking the standard schema)
    # Validate sources expects 'grounded', 'answer'. We adapt parsed:
    adapter = {
        "grounded": parsed.get("comparable", False),
        "answer": "Comparison generated.",
        "used_sources": parsed.get("used_sources", []),
        "limitations": parsed.get("missing_information", [])
    }
    
    validated = validate_sources(adapter, chunks)

    # 5. Build final response
    return {
        "status": (
            "partial_success"
            if parsed.get("comparable") and parsed.get("fallback_used")
            else "success"
            if parsed.get("comparable")
            else "no_context"
        ),
        "question": question,
        "common_points": parsed.get("common_points", []),
        "differences": parsed.get("differences", []),
        "contradictions": parsed.get("contradictions", []),
        "missing_information": validated.get("limitations", []),
        "sources": validated.get("sources", []),
        "confidence": validated.get("confidence", 0.0),
        "fallback_used": bool(parsed.get("fallback_used", False)),
    }
