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
    # Group by filename
    groups = {}
    for c in retrieved_chunks:
        fname = c.get("filename", "unknown")
        groups.setdefault(fname, []).append(c)

    lines = []
    idx = 1
    # Assign global chunk numbering so citations work the same way as standard RAG
    # We update the original list dicts with 'rank' if we want, but numbering here is enough.
    for fname, chunks in groups.items():
        lines.append(f"=== Document: {fname} ===")
        for chunk in chunks:
            # We save this assigned index into the chunk dict temporarily to match later if needed,
            # but standard source validator works by filename. 
            chunk["prompt_idx"] = idx
            header = f"[{idx}] Chunk {chunk.get('chunk_id', '?')} | ReRank={chunk.get('rerank_score', 0.0):.3f}"
            lines.append(f"{header}\n{chunk.get('text', '').strip()}\n")
            idx += 1
    
    return "\n".join(lines)


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
    raw = call_llm(prompt=user_prompt, system=_COMPARATOR_PROMPT, timeout=90, model_name=model_name)
    parsed = safe_json_parse(raw)

    if not parsed:
        parsed = {
            "comparable": False,
            "missing_information": ["LLM parse error."],
            "used_sources": []
        }

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
        "status": "success" if parsed.get("comparable") else "no_context",
        "question": question,
        "common_points": parsed.get("common_points", []),
        "differences": parsed.get("differences", []),
        "contradictions": parsed.get("contradictions", []),
        "missing_information": validated.get("limitations", []),
        "sources": validated.get("sources", []),
        "confidence": validated.get("confidence", 0.0)
    }
