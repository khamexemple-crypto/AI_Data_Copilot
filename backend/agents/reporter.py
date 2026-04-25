"""
Reporter Agent — synthétise tous les outputs en réponse finale utilisateur.
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_reporter_prompt



def run_reporter(user_query: str, plan: dict, analysis: dict, critic: dict, model_name: str = None) -> dict:
    """
    Appelle le LLM pour produire la synthèse finale.
    Retourne toujours un dict valide.
    """
    print("📝 [Reporter] Synthèse finale en cours...")

    system, user = build_reporter_prompt(user_query, plan, analysis, critic)

    raw = call_llm(prompt=user, system=system, timeout=120, model_name=model_name)
    parsed = safe_json_parse(raw)

    if parsed and "final_answer" in parsed:
        return {
            "summary": parsed.get("summary", ""),
            "final_answer": parsed.get("final_answer", "")
        }
    else:
        # Si on ne peut pas parser, on utilise le raw au lieu de throw si possible ?
        # Non, on doit throw pour utiliser le fallback standard
        raise ValueError("JSON invalide ou champs manquants dans la réponse du Reporter.")
