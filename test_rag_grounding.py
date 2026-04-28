"""
Smoke test for the grounded RAG stack.
Run from project root: python test_rag_grounding.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backend.rag.source_validator import (
    detect_insufficient_context,
    compute_confidence,
    detect_contradictions,
    build_source_citations,
    validate_sources,
)
from backend.prompts import build_rag_prompt

PASS = "✅"
FAIL = "❌"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}  ← {detail}")
        errors.append(label)

# ── Test data ─────────────────────────────────────────────────────────────────
good_chunks = [
    {
        "chunk_id": "c1", "file_id": "f1", "filename": "rapport.pdf",
        "text": "L'objectif principal est de démocratiser l'analyse de données et réduire les coûts de 30%.",
        "final_score": 0.75, "rerank_score": 0.62, "rank": 1,
    },
    {
        "chunk_id": "c2", "file_id": "f2", "filename": "business_plan.pdf",
        "text": "Le plan financier prévoit une levée de fonds de 500k€ en 2025 selon les projections.",
        "final_score": 0.55, "rerank_score": 0.41, "rank": 2,
    },
]

weak_chunks = [
    {
        "chunk_id": "w1", "file_id": "f1", "filename": "vague.pdf",
        "text": "Ok.",   # too short
        "final_score": 0.05, "rerank_score": 0.03, "rank": 1,
    },
]

# ── 1. detect_insufficient_context ───────────────────────────────────────────
insuff_empty, _ = detect_insufficient_context([])
check("empty list → insufficient", insuff_empty)

insuff_weak, reason_weak = detect_insufficient_context(weak_chunks)
check("weak chunks → insufficient", insuff_weak, reason_weak)

insuff_good, _ = detect_insufficient_context(good_chunks)
check("good chunks → sufficient", not insuff_good)

# ── 2. compute_confidence ─────────────────────────────────────────────────────
conf_none   = compute_confidence([], llm_grounded=True)
check("no chunks → confidence 0.0", conf_none == 0.0, f"got {conf_none}")

conf_ungrounded = compute_confidence(good_chunks, llm_grounded=False)
check("ungrounded → confidence 0.0", conf_ungrounded == 0.0, f"got {conf_ungrounded}")

conf_good = compute_confidence(good_chunks, llm_grounded=True)
check("good chunks → confidence > 0", conf_good > 0.0, f"got {conf_good}")
check("good chunks → confidence ≤ 1", conf_good <= 1.0, f"got {conf_good}")
print(f"   → computed confidence: {conf_good}")

# ── 3. detect_contradictions ─────────────────────────────────────────────────
contra = detect_contradictions(good_chunks)   # two different files
check("two files → contradiction advisory", len(contra) > 0)

no_contra = detect_contradictions([good_chunks[0]])  # single file
check("single file → no contradiction", len(no_contra) == 0)

# ── 4. build_source_citations ─────────────────────────────────────────────────
citations_filtered = build_source_citations(good_chunks, used_filenames=["rapport.pdf"])
check("filter by filename", len(citations_filtered) == 1)
check("correct filename", citations_filtered[0]["filename"] == "rapport.pdf")
check("excerpt present",   "excerpt" in citations_filtered[0])
check("excerpt not empty", len(citations_filtered[0]["excerpt"]) > 0)

citations_all = build_source_citations(good_chunks, used_filenames=[])
check("no filter → all chunks", len(citations_all) == 2)

# ── 5. build_rag_prompt ───────────────────────────────────────────────────────
system, user = build_rag_prompt("Quels sont les objectifs ?", "[1] some chunk text")
check("system prompt has ABSOLUTE RULES", "ABSOLUTE RULES" in system)
check("system prompt has grounded=false instruction", "grounded" in system)
check("user prompt contains question", "Quels sont les objectifs" in user)
check("user prompt contains context", "[1] some chunk text" in user)

# ── 6. validate_sources — grounded path ──────────────────────────────────────
llm_ok = {
    "answer": "L'objectif est de démocratiser l'analyse selon [1].",
    "used_sources": ["rapport.pdf"],
    "limitations": [],
    "grounded": True,
}
result_ok = validate_sources(llm_ok, good_chunks)
check("grounded → confidence > 0",  result_ok["confidence"] > 0.0)
check("grounded → sources filled",  len(result_ok["sources"]) > 0)
check("source has excerpt",         "excerpt" in result_ok["sources"][0])
check("source filename correct",    result_ok["sources"][0]["filename"] == "rapport.pdf")

# ── 7. validate_sources — ungrounded path ────────────────────────────────────
llm_no = {
    "answer": "",
    "used_sources": [],
    "limitations": [],
    "grounded": False,
}
result_no = validate_sources(llm_no, good_chunks)
check("ungrounded → confidence 0.0",      result_no["confidence"] == 0.0)
check("ungrounded → sources empty",       result_no["sources"] == [])
check("ungrounded → fallback answer set", "Je n'ai pas trouvé" in result_no["answer"])

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"❌ {len(errors)} test(s) FAILED: {errors}")
    sys.exit(1)
else:
    print("🎉 All smoke tests passed — grounded RAG stack is healthy.")
