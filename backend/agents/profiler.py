from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def profiler_agent(state: AgentState) -> dict:
    """
    Profiler Agent: Génère automatiquement un rapport de santé des données (EDA initial).
    """
    llm = get_llm()
    metadata = state["dataset_metadata"]
    
    system_prompt = f"""
    Tu es un Expert Data Profiler. 
    Ta mission est d'étudier les métadonnées d'un nouveau dataset et de définir les étapes pour un "Résumé de Santé" complet.
    
    Dataset Metadata: {metadata}
    
    Tâche :
    - Identifie les problèmes potentiels (valeurs manquantes, types incohérents).
    - Suggère 3 analyses clés (ex: distribution de la cible, corrélations majeures).
    - Formule des instructions pour le Codeur afin de générer ces statistiques de base.
    
    Réponds de manière concise. Ne génère pas de code.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Génère un profil complet pour ce nouveau dataset.")
    ]
    
    response = llm.invoke(messages)
    
    trace = state.get("agent_trace", [])
    trace.append("Profiler (Autonomous Audit)")
    
    return {
        "active_agent": "profiler",
        "agent_thought": response.content,
        "agent_trace": trace,
        "current_plan": ["Audit de santé", "Calcul des stats descriptives", "Top insights"]
    }
