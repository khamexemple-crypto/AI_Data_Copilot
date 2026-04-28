from backend.rag.hybrid_search import run_hybrid_search
from backend.rag.reranker import rerank_chunks

def retrieve_context(query: str, file_ids: list[str] = None):
    # 1. Broad retrieval (Hybrid: Vector + BM25)
    candidates = run_hybrid_search(query, top_k=20, file_ids=file_ids)
    
    if not candidates:
        return {"chunks": [], "status": "no_context"}
        
    # 2. Strict reranking
    best_chunks = rerank_chunks(query, candidates, top_k=5)
    
    if not best_chunks:
        return {"chunks": [], "status": "no_context"}

    # 3. Assess quality based on the top rerank score
    highest_score = best_chunks[0]["rerank_score"]
    status = "weak_context" if highest_score < 0.25 else "strong_context"
    
    return {"chunks": best_chunks, "status": status}
