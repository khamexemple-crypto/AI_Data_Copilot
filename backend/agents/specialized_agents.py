from backend.agents.state import AgentState

def data_analyst_agent(state: AgentState) -> dict:
    """
    Agent dédié à l'analyse de données (EDA, Profiling, etc).
    Pour le moment, il délègue au planificateur ou génère un prompt spécialisé.
    """
    print("📊 [DataAnalyst] Prise en charge de la requête...")
    
    # Pour l'instant, on pré-remplit le plan pour déléguer l'analyse au Coder classique
    # Dans la prochaine itération, cet agent génèrera sa propre réponse sans passer par le coder/executor
    # ou utilisera un outil de profiling dédié.
    plan = [
        "1. Générer un résumé statistique complet du DataFrame.",
        "2. Formater le résultat proprement avec .to_markdown().",
        "3. Stocker dans 'result'."
    ]
    
    return {
        "current_plan": plan,
        "active_agent": "data_analyst",
        # Hack temporaire pour réutiliser la chaîne de code
        "next_node": "coder" 
    }

def visualization_agent(state: AgentState) -> dict:
    """
    Agent dédié à la création de graphiques spécialisés avec Plotly.
    """
    print("📈 [Visualization] Prise en charge de la requête...")
    plan = [
        "1. Générer le graphique Plotly Express demandé.",
        "2. Exporter le json via fig.to_json() dans 'plot_json'.",
        "3. Assigner l'explication à 'result'."
    ]
    
    return {
        "current_plan": plan,
        "active_agent": "visualization",
        "next_node": "coder"
    }
