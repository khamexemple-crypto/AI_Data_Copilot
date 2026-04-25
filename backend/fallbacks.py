"""
Définition des réponses de repli (fallbacks) en cas d'échec du LLM.
"""

PLANNER_FALLBACK = {
    "task_type": "analysis",
    "steps": [
        "Analyser la structure du dataset",
        "Identifier les colonnes importantes",
        "Détecter les anomalies et corrélations",
        "Produire un résumé"
    ],
    "reasoning_summary": "Plan par défaut appliqué suite à une erreur ou un timeout du LLM."
}

ANALYST_FALLBACK = {
    "insights": ["Impossible de générer des insights spécifiques suite à un timeout."],
    "anomalies": ["Analyse des anomalies indisponible."],
    "correlations": ["Analyse des corrélations indisponible."],
    "important_columns": []
}

REVIEWER_FALLBACK = {
    "issues": ["Validation non effectuée à cause d'un timeout."],
    "limitations": ["La qualité de l'analyse n'a pas pu être garantie (timeout)."],
    "confidence": 0.3
}

REPORTER_FALLBACK = {
    "summary": "Analyse partielle en raison d'une erreur de génération.",
    "final_answer": "Désolé, le modèle d'intelligence artificielle a pris trop de temps pour répondre ou était indisponible. Voici une réponse partielle de secours."
}

FAST_FALLBACK = {
    "plan": {
        "task_type": "fast_analysis",
        "steps": ["Fallback rapide"],
        "reasoning_summary": "Erreur lors de l'analyse rapide."
    },
    "analysis": ANALYST_FALLBACK,
    "critic": REVIEWER_FALLBACK,
    "report": REPORTER_FALLBACK
}
