"""
Analyst Agent — analyse les métadonnées et produit des insights structurés.
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_analyst_prompt



def run_analyst(user_query: str, metadata: dict, sample_rows: list, plan: dict, model_name: str = None) -> dict:
    """
    Appelle le LLM pour analyser le dataset.
    Retourne toujours un dict valide.
    """
    print("🔬 [Analyst] Analyse en cours...")

    system, user = build_analyst_prompt(user_query, metadata, sample_rows, plan)

    raw = call_llm(prompt=user, system=system, timeout=120, model_name=model_name)
    parsed = safe_json_parse(raw)

    if parsed and "insights" in parsed:
        # Garantir que toutes les clés attendues sont présentes
        return {
            "insights": parsed.get("insights", []),
            "anomalies": parsed.get("anomalies", []),
            "correlations": parsed.get("correlations", []),
            "important_columns": parsed.get("important_columns", [])
        }
    else:
        raise ValueError("JSON invalide ou champs manquants dans la réponse de l'Analyst.")
