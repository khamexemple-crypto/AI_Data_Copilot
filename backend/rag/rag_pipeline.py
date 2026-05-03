"""
backend/rag/rag_pipeline.py
---------------------------
Orchestrates the full RAG pipeline:

  Question
    → Hybrid Search (top 20 candidates)
    → Reranker       (top 5 best chunks)
    → RAG Agent      (grounded generation)
    → Answer with sources + confidence

No new dependencies required.
"""

import logging
from backend.rag.retriever import retrieve_context
from backend.agents.rag_agent import run_rag_agent

logger = logging.getLogger(__name__)

# Below this rerank_score the pipeline adds a quality warning in the answer
WEAK_CONTEXT_THRESHOLD = 0.25


def execute_rag(query: str, model_name: str = None, file_ids: list[str] = None) -> dict:
    """
    Execute the complete RAG pipeline and return a structured answer dict.

    Returns
    -------
    dict with keys:
        answer       str
        sources      list[dict]   — reranked chunks actually used
        confidence   float        — computed by source_validator
        grounded     bool
        status       str          — "success" | "no_context" | "weak_context"
        question     str
        limitations  list[str]    — optional warnings
        rerank_meta  dict         — top rerank score + num chunks for observability
    """
    # ── Step 1 + 2: Broad Hybrid Retrieval → Reranking ───────────────────────
    logger.info("🔍 RAG pipeline: retrieving + reranking for query: %s", query[:80])
    try:
        retrieval = retrieve_context(query, file_ids=file_ids)          # returns {"chunks": [...], "status": str}
    except Exception as e:
        logger.exception("RAG pipeline: retrieval failed")
        return {
            "answer": "La recherche dans l'index documentaire a échoué.",
            "sources": [],
            "confidence": 0.0,
            "grounded": False,
            "status": "retrieval_error",
            "question": query,
            "limitations": [f"Retrieval failed: {e}"],
            "rerank_meta": {
                "num_chunks_sent_to_llm": 0,
                "top_rerank_score": 0.0,
                "retrieval_status": "retrieval_error",
            },
        }

    retrieved_chunks = retrieval.get("chunks", [])
    retrieval_status = retrieval.get("status", "no_context")

    # Gather rerank observability metadata (safe even if list is empty)
    top_rerank_score = retrieved_chunks[0]["rerank_score"] if retrieved_chunks else 0.0
    rerank_meta = {
        "num_chunks_sent_to_llm": len(retrieved_chunks),
        "top_rerank_score"      : top_rerank_score,
        "retrieval_status"      : retrieval_status,
    }

    if not retrieved_chunks:
        logger.warning("⚠️  RAG pipeline: no chunks retrieved.")
        return {
            "answer"     : "Je n'ai pas trouvé assez d'informations dans les fichiers stockés pour répondre avec certitude.",
            "sources"    : [],
            "confidence" : 0.0,
            "grounded"   : False,
            "status"     : "no_context",
            "question"   : query,
            "limitations": ["Aucun contexte pertinent trouvé dans les documents indexés."],
            "rerank_meta": rerank_meta,
        }

    # ── Step 3 + 4: Grounded Generation → Validation ─────────────────────────
    logger.info(
        "✅ RAG pipeline: sending %d reranked chunks to LLM (top score=%.3f)",
        len(retrieved_chunks), top_rerank_score,
    )
    answer_dict = run_rag_agent(query, retrieved_chunks, model_name)

    # ── Step 5: Build final response ──────────────────────────────────────────
    answer_dict["question"]    = query
    answer_dict["rerank_meta"] = rerank_meta

    if answer_dict.get("llm_error"):
        answer_dict["status"] = "llm_error"
    elif answer_dict.get("grounded"):
        answer_dict["status"] = "success"
    else:
        answer_dict["status"] = "no_context"

    # Propagate weak-context warning even when LLM claims it's grounded
    if retrieval_status == "weak_context":
        answer_dict["status"] = "weak_context"
        answer_dict.setdefault("limitations", []).append(
            f"⚠️ Le contexte récupéré est faible (top rerank score: {top_rerank_score:.2f}). "
            "La réponse peut manquer de précision."
        )
        logger.warning(
            "⚠️  RAG pipeline: weak context — top rerank score=%.3f", top_rerank_score
        )

    return answer_dict
