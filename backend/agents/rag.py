from backend.agents.state import AgentState
from backend.core.retrieval import retrieve_hybrid_context

def rag_agent(state: AgentState) -> dict:
    """
    L'agent RAG récupère du contexte pertinent par rapport à la requête utilisateur.
    Ce contexte est ajouté aux messages système pour les agents suivants.
    """
    print("🖇️ [RAG Agent] Récupération de contexte externe/interne...")
    session_id = state["session_id"]
    query = state["user_prompt"]
    
    # Récupération du contexte via le moteur hybride
    context = retrieve_hybrid_context(session_id, query)
    
    if not context:
        print("🖇️ [RAG Agent] Aucun contexte supplémentaire trouvé.")
        return {"active_agent": "rag", "next_agent": "planner"}
        
    print(f"🖇️ [RAG Agent] Contexte récupéré ({len(context)} caractères).")
    
    # On ajoute le contexte récupéré au prompt utilisateur pour que l'agent suivant (ex: DataAnalyst ou Planner) en profite
    # On peut aussi le mettre dans un champ spécifique "rag_context" de l'état si on veut être propre
    new_messages = state.get("messages", [])
    new_messages.append({
        "role": "system", 
        "content": f"CONTEXTE RÉCUPÉRÉ (RAG) :\n{context}\n\nUtilise ces informations pour répondre de manière plus précise."
    })
    
    return {
        "messages": new_messages,
        "active_agent": "rag",
        "next_agent": "supervisor" # On repasse par le superviseur pour décider de la suite avec le nouveau contexte
    }
