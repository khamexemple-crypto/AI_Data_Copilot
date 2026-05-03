from backend.rag.vector_store import search_store as vector_search
from backend.rag.keyword_index import keyword_store
import logging

logger = logging.getLogger(__name__)

def normalize_scores(results: list, key="score"):
    if not results: return []
    scores = [r.get(key, 0.0) for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results: r["norm_score"] = 1.0
        return results
    for r in results:
        r["norm_score"] = (r.get(key, 0.0) - min_s) / (max_s - min_s)
    return results

def run_hybrid_search(query: str, top_k: int = 20, file_ids: list[str] = None):
    """
    Hybrid search: Vector (Chroma) + BM25 keyword.
    Returns up to *top_k* merged, deduplicated chunks sorted by final_score.
    Default bumped to 20 to give the reranker enough candidates.
    """
    # 1. Get raw results — fetch top_k from each sub-index so the
    #    merged pool contains at least top_k unique candidates.
    try:
        v_results = vector_search(query, n_results=top_k, file_ids=file_ids)
    except Exception as e:
        logger.warning("Hybrid search: vector search failed, continuing with keyword search: %s", e)
        v_results = []

    # Keyword results (higher is better)
    try:
        k_results = keyword_store.search(query, top_k=top_k, file_ids=file_ids)
    except Exception as e:
        logger.warning("Hybrid search: keyword search failed, continuing with vector search: %s", e)
        k_results = []
    
    # 2. Normalize
    # Convert Chroma distance to similarity (1 / (1 + distance))
    for v in v_results: 
        v['score'] = 1.0 / (1.0 + v.get('distance', 1.0))
    
    v_results = normalize_scores(v_results, key="score")
    k_results = normalize_scores(k_results, key="score")
    
    # 3. Merge & Deduplicate
    merged = {}
    
    for v in v_results:
        cid = f"{v['file_id']}_{v['chunk_id']}"
        v["final_score"] = 0.7 * v["norm_score"]
        v["search_type"] = "vector"
        merged[cid] = v
        
    for k in k_results:
        cid = f"{k['file_id']}_{k['chunk_id']}"
        if cid in merged:
            merged[cid]["final_score"] += 0.3 * k["norm_score"]
            merged[cid]["search_type"] = "hybrid"
        else:
            k["final_score"] = 0.3 * k["norm_score"]
            k["search_type"] = "keyword"
            merged[cid] = k
            
    # 4. Sort and return top_k
    sorted_chunks = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
    return sorted_chunks[:top_k]
