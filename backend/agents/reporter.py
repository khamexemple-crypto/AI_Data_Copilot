import json
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def reporter_agent(state: AgentState) -> dict:
    """
    Reporter Agent: Consolide tous les résultats des agents en un format JSON structuré.
    """
    llm = get_llm()
    
    # On rassemble tout ce qu'on a collecté
    context = {
        "user_prompt": state["user_prompt"],
        "plan": state.get("current_plan", []),
        "insights": state.get("insights", []),
        "anomalies": state.get("anomalies", []),
        "correlations": state.get("correlations", []),
        "execution_result": state.get("execution_result"),
        "forecast_results": state.get("forecast_results", {}),
        "critique": state.get("critique", {})
    }
    
    system_prompt = """
    Tu es le Rapporteur Final du système multi-agents.
    Ton rôle est de prendre tous les éléments d'analyse fournis et de produire une réponse finale cohérente et TRÈS STRUCTURÉE au format JSON.
    
    Tu dois impérativement respecter ce schéma JSON :
    {
        "plan": ["étape 1", "..."],
        "analysis": {
            "insights": ["insight 1", "..."],
            "anomalies": ["anomalie 1", "..."],
            "correlations": ["correlation 1", "..."]
        },
        "predictions": {
            "applied": true/false,
            "method": "nom méthode",
            "result_summary": "résumé prédictif"
        },
        "critic": {
            "issues": ["problème 1", "..."],
            "confidence": 0.0,
            "limitations": ["limitation 1", "..."]
        },
        "final_answer": "Texte clair et pédagogique résumant toute l'analyse pour l'utilisateur."
    }
    
    N'ajoute aucun texte avant ou après le JSON.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Contexte accumulé : {json.dumps(context, default=str)}")
    ]
    
    print("📝 [Reporter] Synthèse finale en cours...")
    response = llm.invoke(messages)
    
    try:
        # Nettoyage si l'LLM a mis des backticks ```json ... ```
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        structured_data = json.loads(content)
        trace = state.get("agent_trace", [])
        trace.append("Reporter (Synthesis)")
        
        return {
            "structured_output": structured_data,
            "final_answer": structured_data.get("final_answer", "Analyse terminée."),
            "agent_trace": trace
        }
    except Exception as e:
        print(f"⚠️ [Reporter] Erreur lors du parsing JSON : {e}")
        trace = state.get("agent_trace", [])
        trace.append("Reporter (Fallback)")
        return {
            "final_answer": response.content, # Fallback text
            "structured_output": {"final_answer": response.content},
            "agent_trace": trace
        }
