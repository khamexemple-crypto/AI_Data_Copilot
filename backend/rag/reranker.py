"""
backend/rag/reranker.py
-----------------------
Option A lightweight reranker (no heavy deps).

Scoring components
──────────────────
  hybrid_score   — normalised Hybrid Search score from retriever    (weight 0.50)
  overlap_score  — query token overlap + in-text frequency density   (weight 0.30)
  exact_bonus    — reward multi-word phrase / exact-term hit          (weight 0.15)
  position_bonus — reward chunks whose first sentence already answers (weight 0.05)

Length penalty   — halves the score for chunks shorter than 50 chars.
Low-score guard  — warns the caller when top chunk is very weak.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────
WEAK_SCORE_THRESHOLD = 0.20   # below this → warn the LLM caller
SHORT_CHUNK_CHARS    = 50     # penalty below this length


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation, removing common stop-words."""
    STOPWORDS = {
        "le", "la", "les", "de", "du", "des", "un", "une", "en", "et",
        "à", "au", "aux", "pour", "par", "sur", "dans", "avec", "que",
        "qui", "se", "ne", "pas", "the", "a", "an", "of", "in", "to",
        "is", "it", "on", "at", "by", "or", "be", "as", "we",
    }
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_keyword_overlap(query: str, text: str) -> float:
    """
    Two-part lexical score:
      1. Query coverage  — fraction of unique query tokens found in the text.
      2. Density bonus   — how frequently those tokens appear (capped at 0.20).

    Returns a value in [0.0, 1.20] (the density bonus can push slightly above 1).
    """
    query_tokens = set(_tokenize(query))
    text_tokens  = _tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)

    # 1. Coverage
    matched       = query_tokens & text_token_set
    coverage      = len(matched) / len(query_tokens)

    # 2. Density bonus — how many times query tokens appear in the text
    freq_count    = sum(1 for t in text_tokens if t in query_tokens)
    density_bonus = min(freq_count / (len(text_tokens) + 1), 0.20)

    return coverage + density_bonus


def _compute_exact_term_bonus(query: str, text: str) -> float:
    """
    Rewards chunks that contain:
      • The full query as a literal sub-string   → +0.30
      • Any bi-gram from the query               → +0.10 per bi-gram (capped at 0.30)
      • Any single query token as a whole word   → +0.03 per token  (capped at 0.15)

    Returns a value in [0.0, 0.60].
    """
    text_lower  = text.lower()
    query_lower = query.lower()
    bonus       = 0.0

    # Full phrase match
    if query_lower in text_lower:
        bonus += 0.30

    # Bi-gram matches
    tokens  = _tokenize(query)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    bigram_hits = sum(1 for bg in bigrams if bg in text_lower)
    bonus += min(bigram_hits * 0.10, 0.30)

    # Single-token whole-word matches (only if no full-phrase match already)
    if bonus < 0.30:
        for token in tokens:
            if re.search(rf'\b{re.escape(token)}\b', text_lower):
                bonus += 0.03
        bonus = min(bonus, 0.15 + 0.30)  # total cap

    return min(bonus, 0.60)


def _compute_position_bonus(query: str, text: str) -> float:
    """
    Small bonus if a query token appears in the first 200 chars of the chunk
    (proxy for the chunk leading with an answer).
    """
    head        = text[:200].lower()
    query_tokens = set(_tokenize(query))
    hits        = sum(1 for t in query_tokens if t in head)
    return min(hits * 0.02, 0.10)


# ── Main scoring function ────────────────────────────────────────────────────

def compute_final_rerank_score(query: str, chunk: dict) -> float:
    """
    Combine all signals into one rerank_score in [0.0, 1.0+].

    Weights:
        hybrid_score   × 0.50
        overlap_score  × 0.30
        exact_bonus    × 0.15  (already bounded ≤ 0.60, scaled into weight)
        position_bonus × 0.05

    A length_penalty of 0.5 halves the final score for very short chunks.
    """
    text          = chunk.get("text", "")
    hybrid_score  = chunk.get("final_score", 0.0)

    # Length penalty
    length_penalty = 0.5 if len(text) < SHORT_CHUNK_CHARS else 1.0

    overlap_score  = compute_keyword_overlap(query, text)           # [0, ~1.2]
    exact_bonus    = _compute_exact_term_bonus(query, text)         # [0,  0.60]
    position_bonus = _compute_position_bonus(query, text)           # [0,  0.10]

    raw_score = (
        hybrid_score  * 0.50 +
        overlap_score * 0.30 +
        exact_bonus   * 0.15 +
        position_bonus * 0.05
    )

    return round(raw_score * length_penalty, 4)


# ── Public entry point ───────────────────────────────────────────────────────

def rerank_chunks(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Rerank *chunks* by relevance to *query* and return the top_k best ones.

    Each returned chunk is guaranteed to contain:
        chunk_id     str
        file_id      str
        filename     str
        text         str
        score        float  (original hybrid score)
        rerank_score float  (new rerank score)
        rank         int    (1 = best)

    Side-effects:
        • Logs a WARNING when the top chunk score is below WEAK_SCORE_THRESHOLD.
    """
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        rerank_score = compute_final_rerank_score(query, chunk)

        scored.append({
            # ── Required schema ──────────────────────────────────────────
            "chunk_id"    : str(chunk.get("chunk_id", "")),
            "file_id"     : str(chunk.get("file_id", "")),
            "filename"    : str(chunk.get("filename", "")),
            "text"        : chunk.get("text", ""),
            # ── Scores ───────────────────────────────────────────────────
            "score"       : round(chunk.get("final_score", 0.0), 4),
            "rerank_score": rerank_score,
            # ── Extra passthrough (search_type, norm_score …) ────────────
            **{k: v for k, v in chunk.items()
               if k not in {"chunk_id", "file_id", "filename", "text",
                             "final_score", "rerank_score", "rank"}},
        })

    # Sort descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Assign rank
    top_chunks = scored[:top_k]
    for idx, c in enumerate(top_chunks):
        c["rank"] = idx + 1

    # Quality guard
    top_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0
    if top_score < WEAK_SCORE_THRESHOLD:
        logger.warning(
            "⚠️  Reranker: top chunk score %.3f is below threshold %.2f. "
            "Answer confidence will be low.",
            top_score, WEAK_SCORE_THRESHOLD,
        )

    return top_chunks
