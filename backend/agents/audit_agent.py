"""
Audit Agent — génère un rapport d'audit global (Dataset + Documents).
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_audit_prompt
import logging

logger = logging.getLogger(__name__)

def run_audit_agent(
    dataset_analysis: dict | None,
    document_intelligence: dict | None,
    model_name: str | None = None
) -> dict:
    """
    Exécute l'Audit Agent pour générer un rapport global.
    Retourne toujours un dictionnaire valide.
    """
    logger.info("🧪 [Audit Agent] Audit en cours...")

    system, user = build_audit_prompt(dataset_analysis, document_intelligence)

    try:
        raw = call_llm(prompt=user, system=system, timeout=120, model_name=model_name)
        parsed = safe_json_parse(raw)

        if parsed and "summary" in parsed:
            return {
                "status": "success",
                "summary": parsed.get("summary", ""),
                "dataset_quality": parsed.get("dataset_quality", []),
                "document_findings": parsed.get("document_findings", []),
                "risks": parsed.get("risks", []),
                "contradictions": parsed.get("contradictions", []),
                "opportunities": parsed.get("opportunities", []),
                "recommendations": parsed.get("recommendations", []),
                "confidence": float(parsed.get("confidence", 0.0)),
                "limitations": parsed.get("limitations", [])
            }
        else:
            raise ValueError("Invalid JSON or missing fields in Audit Agent response.")
    except Exception as e:
        logger.error(f"Audit Agent error: {e}")
        return {
            "status": "error",
            "summary": "L'audit a échoué en raison d'une erreur interne.",
            "dataset_quality": [],
            "document_findings": [],
            "risks": [],
            "contradictions": [],
            "opportunities": [],
            "recommendations": [],
            "confidence": 0.0,
            "limitations": [str(e)]
        }
