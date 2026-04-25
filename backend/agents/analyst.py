from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def analyst_node(state: AgentState) -> dict:
    """
    Analyst Agent: Analyse la problématique et prépare les instructions de code pour le Coder.
    """
    llm = get_llm()
    metadata = state["dataset_metadata"]
    user_prompt = state["user_prompt"]
    
    system_prompt = f"""
    Tu es un Expert Data Analyst. 
    Ton rôle est d'analyser la demande et de formuler des instructions PRÉCISES pour un développeur Python/Pandas.
    
    Dataset Metadata: {metadata}
    
    Tâche :
    - Détermine les calculs statistiques nécessaires (moyenne, corrélations, etc.).
    - Identifie les colonnes à utiliser.
    - Suggère des méthodes de nettoyage si besoin.
    
    Réponds de manière concise. Ne génère PAS de code.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Demande utilisateur: {user_prompt}")
    ]
    
    response = llm.invoke(messages)
    trace = state.get("agent_trace", [])
    trace.append("Analyst (Reasoning)")
    return {
        "active_agent": "analyst",
        "agent_thought": response.content,
        "agent_trace": trace
    }
