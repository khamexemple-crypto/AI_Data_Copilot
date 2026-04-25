from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def planner_agent(state: AgentState) -> dict:
    """
    Le Planner Agent: Décompose la requête utilisateur en étapes d'analyse.
    """
    llm = get_llm()
    prompt = state["user_prompt"]
    metadata = state["dataset_metadata"]
    
    system_prompt = f"""
    Tu es le Cerveau Stratégique d'une plateforme d'IA Data Science.
    Ton rôle est de créer un plan d'action séquentiel pour répondre à l'utilisateur.
    
    Dataset Metadata: {metadata}
    
    Types de spécialistes disponibles :
    1. Analyst: Pour les calculs statistiques, corrélations, EDA.
    2. Visualization: Pour générer des graphiques Plotly.
    3. Forecasting: Pour des prédictions/séries temporelles.
    
    Réponds au format JSON suivant :
    {{
        "plan": ["étape 1", "étape 2", "..."],
        "first_step": "Analyst" | "Visualization" | "Forecasting",
        "reasoning": "Pourquoi ce plan ?"
    }}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        # Nettoyage si besoin
        res_text = response.content.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:-3].strip()
            
        data = json.loads(res_text)
        plan = data.get("plan", [])
        next_agent = data.get("first_step", "analyst").lower()
        
        trace = state.get("agent_trace", [])
        trace.append("Planner")
        return {
            "current_plan": plan,
            "next_agent": next_agent,
            "active_agent": "planner",
            "agent_trace": trace
        }
    except Exception as e:
        print(f"⚠️ [Planner] Erreur : {e}. Fallback vers analyst.")
        trace = state.get("agent_trace", [])
        trace.append("Planner (Error Fallback)")
        return {
            "current_plan": ["Analyser les données"],
            "next_agent": "analyst",
            "active_agent": "planner",
            "agent_trace": trace
        }
