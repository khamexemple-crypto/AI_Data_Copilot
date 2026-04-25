from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def forecasting_agent(state: AgentState) -> dict:
    """
    Forecasting Agent: Détermine la meilleure approche prédictive.
    """
    llm = get_llm()
    metadata = state["dataset_metadata"]
    user_prompt = state["user_prompt"]
    
    system_prompt = f"""
    Tu es un Expert en Machine Learning.
    Ton rôle est de définir une stratégie de modélisation pour répondre à la demande de l'utilisateur.
    
    Dataset Metadata: {metadata}
    
    Tâche :
    - Détermine si une prédiction est possible (présence de colonnes temporelles ou cibles).
    - Sélectionne un modèle (Régression, Prophet, etc.).
    - Donne des instructions techniques précises pour le Codeur.
    
    Réponds de manière concise sans code.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Demande: {user_prompt}")
    ]
    
    response = llm.invoke(messages)
    trace = state.get("agent_trace", [])
    trace.append("Forecaster (Prediction)")
    return {
        "active_agent": "forecaster",
        "agent_thought": response.content,
        "agent_trace": trace
    }
