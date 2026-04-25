"""
Fast Agent — exécute une passe rapide pour obtenir un résumé direct.
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_fast_prompt



def run_fast_agent(user_query: str, metadata: dict, sample_rows: list, model_name: str = None) -> dict:
    """
    Appelle le LLM pour faire une passe rapide complète.
    """
    print("⚡ [Fast Agent] Analyse rapide en cours...")

    system, user = build_fast_prompt(user_query, metadata, sample_rows)

    raw = call_llm(prompt=user, system=system, timeout=60, model_name=model_name)
    parsed = safe_json_parse(raw)

    if parsed and "report" in parsed:
        return parsed
    else:
        raise ValueError("JSON invalide ou champs manquants dans la réponse du Fast Agent.")
