"""
Planner Agent — détecte le type de tâche et produit un plan d'action.
"""

from backend.llm import call_llm, safe_json_parse
from backend.prompts import build_planner_prompt




def run_planner(user_query: str, metadata: dict, sample_rows: list, model_name: str = None) -> dict:
    """
    Appelle le LLM pour planifier la réponse à la requête utilisateur.
    Retourne toujours un dict valide, même en cas d'échec.
    """
    print("🗂️  [Planner] Planification en cours...")

    system, user = build_planner_prompt(user_query, metadata, sample_rows)

    raw = call_llm(prompt=user, system=system, timeout=90, model_name=model_name)
    parsed = safe_json_parse(raw)

    if parsed and "task_type" in parsed and "steps" in parsed:
        print(f"✅ [Planner] task_type={parsed['task_type']}")
        return parsed
    else:
        raise ValueError("JSON invalide ou champs manquants dans la réponse du Planner.")
