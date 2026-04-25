from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def visualization_node(state: AgentState) -> dict:
    """
    Visualization Agent: Détermine la meilleure stratégie visuelle.
    """
    llm = get_llm()
    metadata = state["dataset_metadata"]
    user_prompt = state["user_prompt"]
    
    system_prompt = f"""
    Tu es un Expert en Visualisation de Données (Storytelling).
    Ton rôle est de choisir le meilleur graphique pour répondre à la demande de l'utilisateur.
    
    Dataset Metadata: {metadata}
    
    Tâche :
    - Sélectionne le type de graphique Plotly Express approprié.
    - Précise les colonnes pour X, Y, Color, etc.
    - Donne des instructions claires pour le Codeur.
    
    Réponds de manière concise sans code.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Demande: {user_prompt}")
    ]
    
    response = llm.invoke(messages)
    trace = state.get("agent_trace", [])
    trace.append("Visualization (Design)")
    return {
        "active_agent": "visualization",
        "agent_thought": response.content,
        "agent_trace": trace
    }
