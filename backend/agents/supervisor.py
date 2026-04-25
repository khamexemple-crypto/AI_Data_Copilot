from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.config import settings
from backend.agents.state import AgentState
import json

llm = None

def get_llm():
    global llm
    if llm is None:
        # On utilise format="json" car le superviseur doit renvoyer un objet JSON strict pour le routage
        llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, format="json")
    return llm

def supervisor_agent(state: AgentState) -> dict:
    """
    Le Superviseur analyse l'intention de l'utilisateur et décide quel agent doit prendre le relais.
    Il renvoie une mise à jour d'état contenant le `next_agent`.
    """
    print("👔 [Supervisor] Analyse de la requête utilisateur...")
    llm_instance = get_llm()
    prompt = state["user_prompt"]
    
    system_prompt = """
    Tu es le Routeur/Superviseur d'un système multi-agents spécialisé en Data Science.
    Ta mission est d'analyser la demande de l'utilisateur et de la router vers l'agent le plus pertinent.
    
    Voici les agents disponibles:
    - "data_analyst" : Pour les demandes de profilage, résumés statistiques, détection d'outliers, compréhension générale des données.
    - "rag" : À utiliser si l'utilisateur pose une question nécessitant du contexte externe, une explication de la structure documentaire, ou s'il semble manquer d'informations.
    - "visualization" : Pour de simples demandes de tracés de graphiques ou visualisations.
    - "ml_agent" : Pour l'entraînement de modèles prédictifs, le Machine Learning ou l'AutoML.
    - "planner" : Agent par défaut pour manipuler les données (nettoyage simple, calculs, transformations, exécution).
    
    Tu DOIS renvoyer un objet JSON valide avec une seule clé 'next_agent' contenant le nom de l'agent choisi : 'data_analyst', 'rag', 'visualization', 'ml_agent', ou 'planner'.
    
    Exemple de réponse attendue:
    {
        "next_agent": "rag"
    }
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm_instance.invoke(messages)
    
    try:
        decision = json.loads(response.content)
        next_agent = decision.get("next_agent", "planner")
    except Exception as e:
        print(f"⚠️ [Supervisor] Erreur de parsing JSON ({e}). Fallback sur 'planner'.")
        next_agent = "planner"
        
    print(f"👔 [Supervisor] Décision de routage prise -> {next_agent}")
    
    return {"next_agent": next_agent, "active_agent": "supervisor"}
