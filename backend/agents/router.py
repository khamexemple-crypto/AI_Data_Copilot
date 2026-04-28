"""
backend/agents/router.py
-------------------------
Router Agent — classifies user requests into one of four routes:

  dataset  — structured data questions (CSV / Excel analysis)
  files    — document questions (PDF / DOCX / TXT via RAG)
  mixed    — requires both dataset analysis AND document knowledge
  general  — no data needed, conceptual / conversational

Design philosophy
─────────────────
  1. Deterministic keyword rules run first (zero LLM cost, near-instant).
  2. If deterministic rules are not confident enough, we fall back to a lightweight
     LLM classification call.
  3. The router NEVER breaks existing /analyze or /ask-files endpoints.
"""

import re
import logging
from backend.llm import call_llm, safe_json_parse

logger = logging.getLogger(__name__)

# ── Route constants ───────────────────────────────────────────────────────────
ROUTE_DATASET = "dataset"
ROUTE_FILES   = "files"
ROUTE_MIXED   = "mixed"
ROUTE_GENERAL = "general"

# ── Keyword rule tables ───────────────────────────────────────────────────────
# Tokens that strongly signal a structured-data (dataset) question
_DATASET_KEYWORDS = {
    # French
    "csv", "colonne", "colonnes", "ligne", "lignes", "dataframe", "tableau",
    "statistique", "statistiques", "distribution", "corrélation", "corrélations",
    "analyse", "analyser", "outlier", "outliers", "anomalie", "anomalies",
    "valeur", "valeurs", "manquante", "manquantes", "profil", "profiling",
    "moyenne", "médiane", "écart-type", "min", "max", "somme", "total",
    "graphique", "visualisation", "prédiction", "prédire", "modèle", "forecast",
    "nettoyage", "nettoyer", "dataset",
    # English
    "column", "columns", "row", "rows", "dataframe", "table",
    "statistic", "statistics", "distribution", "correlation", "correlations",
    "analysis", "analyze", "analyse", "outlier", "anomaly", "missing",
    "prediction", "predict", "model", "forecast", "cleaning", "clean",
    "visualization", "visualize", "chart", "plot",
}

# Tokens that strongly signal a document (files/RAG) question
_FILES_KEYWORDS = {
    # French
    "pdf", "document", "fichier", "fichiers", "rapport", "rapports",
    "contrat", "contrats", "cahier des charges", "cahier", "charges",
    "page", "pages", "paragraphe", "paragraphes", "texte", "extraire",
    "résumé", "résumer", "synthèse", "synthétiser", "mentionné", "mentionnée",
    "décrit", "décrite", "stipule", "précise", "indique", "selon le",
    "d'après le", "dans le", "dans les", "objectif", "objectifs",
    "exigence", "exigences", "spécification", "spécifications",
    "business plan", "business_plan",
    # English
    "document", "file", "files", "pdf", "docx", "txt", "report",
    "contract", "specification", "requirements", "summary", "summarize",
    "extract", "according to", "mentioned", "described", "stated", "page",
}

# Tokens that trigger mixed routing (dataset + docs together)
_MIXED_KEYWORDS = {
    # French
    "compare", "comparer", "confronter", "croiser", "rapprocher",
    "vs", "versus", "par rapport", "par rapport à", "objectif",
    "avec les", "avec le", "en ligne avec", "conformément",
    # English
    "compare", "comparison", "versus", "vs", "against", "align",
    "cross-reference", "benchmark",
}


def _tokenize_lower(text: str) -> set[str]:
    """Lowercase word tokenisation — keep multi-word phrases as bigrams."""
    tokens = set(re.findall(r'\w+', text.lower()))
    # Add bigrams (sliding window of 2)
    words = re.findall(r'\w+', text.lower())
    bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
    return tokens | bigrams


def _deterministic_route(question: str) -> tuple[str | None, float, str]:
    """
    Apply keyword rules.

    Returns (route | None, confidence, reason).
    Returns None if rules are not decisive (triggers LLM fallback).
    """
    tokens = _tokenize_lower(question)

    hits_dataset = len(tokens & _DATASET_KEYWORDS)
    hits_files   = len(tokens & _FILES_KEYWORDS)
    hits_mixed   = len(tokens & _MIXED_KEYWORDS)

    # Mixed: explicit comparison keyword + at least one signal from each side
    if hits_mixed >= 1 and hits_dataset >= 1 and hits_files >= 1:
        return (
            ROUTE_MIXED,
            0.88,
            f"Mixed signals: {hits_mixed} comparison keyword(s), "
            f"{hits_dataset} dataset keyword(s), {hits_files} file keyword(s).",
        )

    # Clear dataset win
    if hits_dataset >= 2 and hits_dataset > hits_files * 2:
        return (
            ROUTE_DATASET,
            0.85,
            f"Dominant dataset keywords ({hits_dataset} hits).",
        )

    # Clear files win
    if hits_files >= 2 and hits_files > hits_dataset * 2:
        return (
            ROUTE_FILES,
            0.85,
            f"Dominant file/document keywords ({hits_files} hits).",
        )

    # Weak single-side signal
    if hits_dataset >= 1 and hits_files == 0:
        return (
            ROUTE_DATASET,
            0.65,
            f"Weak dataset signal ({hits_dataset} hit).",
        )

    if hits_files >= 1 and hits_dataset == 0:
        return (
            ROUTE_FILES,
            0.65,
            f"Weak file signal ({hits_files} hit).",
        )

    # Tie or no signal — defer to LLM
    return None, 0.0, "No decisive keyword signal."


def _llm_route(question: str, model_name: str | None = None) -> dict:
    """
    Use a lightweight LLM call to classify the route when keyword rules fail.
    """
    from backend.prompts import build_router_prompt
    system, user = build_router_prompt(question)

    try:
        raw    = call_llm(prompt=user, system=system, timeout=30, model_name=model_name)
        parsed = safe_json_parse(raw)

        if parsed and parsed.get("route") in {ROUTE_DATASET, ROUTE_FILES, ROUTE_MIXED, ROUTE_GENERAL}:
            return {
                "route"           : parsed["route"],
                "reason"          : parsed.get("reason", "LLM classification."),
                "required_agents" : parsed.get("required_agents", []),
                "confidence"      : float(parsed.get("confidence", 0.6)),
                "method"          : "llm",
            }
    except Exception as e:
        logger.warning("Router LLM fallback failed: %s", e)

    # Hard fallback
    return {
        "route"           : ROUTE_GENERAL,
        "reason"          : "Router failed — defaulting to general.",
        "required_agents" : [],
        "confidence"      : 0.0,
        "method"          : "fallback",
    }


# ── Agent map ─────────────────────────────────────────────────────────────────

_ROUTE_AGENTS = {
    ROUTE_DATASET: ["Planner Agent", "Analyst Agent", "Reviewer Agent", "Reporter Agent"],
    ROUTE_FILES  : ["RAG Agent"],
    ROUTE_MIXED  : ["Planner Agent", "Analyst Agent", "RAG Agent", "Reporter Agent"],
    ROUTE_GENERAL: [],
}


# ── Public API ────────────────────────────────────────────────────────────────

def route_question(
    question: str,
    has_session: bool = False,
    has_files: bool = False,
    model_name: str | None = None,
) -> dict:
    """
    Classify *question* into a route and return a routing decision dict.

    Parameters
    ──────────
    question    : raw user question
    has_session : True if a dataset session is active in memory
    has_files   : True if the file knowledge base has indexed documents
    model_name  : optional LLM override for the LLM fallback path

    Returns
    ───────
    dict:
        route            str   — dataset | files | mixed | general
        reason           str   — human-readable explanation
        required_agents  list  — agents that will be invoked
        confidence       float — 0.0–1.0
        method           str   — "rules" | "llm" | "context" | "fallback"
    """
    question = question.strip()

    # ── Context-aware shortcuts ────────────────────────────────────────────────
    # If the user has NO session AND NO files, we can shortcut immediately.
    if not has_session and not has_files:
        logger.info("router → general (no session, no files)")
        return {
            "route"           : ROUTE_GENERAL,
            "reason"          : "No dataset session and no indexed files available.",
            "required_agents" : [],
            "confidence"      : 0.95,
            "method"          : "context",
        }

    # If only one side is available, bias routing toward it
    available_routes = set()
    if has_session : available_routes.add(ROUTE_DATASET)
    if has_files   : available_routes.add(ROUTE_FILES)

    # ── Deterministic rules ────────────────────────────────────────────────────
    det_route, det_conf, det_reason = _deterministic_route(question)

    if det_route and det_conf >= 0.70:
        # Degrade to general if the determined route is unavailable
        if det_route == ROUTE_DATASET and not has_session:
            det_route, det_reason = (
                ROUTE_GENERAL,
                "Route was 'dataset' but no session is active.",
            )
        elif det_route == ROUTE_FILES and not has_files:
            det_route, det_reason = (
                ROUTE_GENERAL,
                "Route was 'files' but no indexed documents found.",
            )

        logger.info("router → %s (rules, conf=%.2f)", det_route, det_conf)
        return {
            "route"           : det_route,
            "reason"          : det_reason,
            "required_agents" : _ROUTE_AGENTS.get(det_route, []),
            "confidence"      : det_conf,
            "method"          : "rules",
        }

    # ── LLM fallback ─────────────────────────────────────────────────────────
    logger.info("router → LLM fallback (det_conf=%.2f)", det_conf)
    result = _llm_route(question, model_name)

    # Same availability degradation for LLM results
    if result["route"] == ROUTE_DATASET and not has_session:
        result["route"]  = ROUTE_GENERAL
        result["reason"] += " (degraded: no active session)"
    elif result["route"] == ROUTE_FILES and not has_files:
        result["route"]  = ROUTE_GENERAL
        result["reason"] += " (degraded: no indexed files)"

    result["required_agents"] = _ROUTE_AGENTS.get(result["route"], [])
    return result
