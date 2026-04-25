"""
SHIM DE COMPATIBILITÉ
Ce fichier redirige les appels de l'ancien orchestrateur LangGraph
vers le nouvel orchestrateur séquentiel (backend/orchestrator.py).

Ne pas supprimer : il est importé par backend/api/routes.py via
  from backend.agents.orchestrator import run_orchestrator
"""

# Réexport du nouvel orchestrateur
from backend.orchestrator import run_orchestrator, run_analyze_pipeline  # noqa: F401
