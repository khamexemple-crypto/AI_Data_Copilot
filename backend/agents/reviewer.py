"""
Reviewer Agent — critique les conclusions de l'Analyst pour limiter les hallucinations.
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_reviewer_prompt



def run_reviewer(user_query: str, analyst_output: dict, metadata: dict, model_name: str = None) -> dict:
    """
    Appelle le LLM pour critiquer l'output de l'Analyst.
    Retourne toujours un dict valide.
    """
    print("🔍 [Reviewer] Révision critique en cours...")

    system, user = build_reviewer_prompt(user_query, analyst_output, metadata)

    raw = call_llm(prompt=user, system=system, timeout=90, model_name=model_name)
    parsed = safe_json_parse(raw)

    if parsed and "confidence" in parsed:
        confidence = float(parsed.get("confidence", 0.5))
        # Clamp entre 0.0 et 1.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "issues": parsed.get("issues", []),
            "limitations": parsed.get("limitations", []),
            "confidence": round(confidence, 2)
        }
    else:
        raise ValueError("JSON invalide ou champs manquants dans la réponse du Reviewer.")
