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
    Tu es un Data Lead expérimenté. 
    L'utilisateur pose une question sur un dataset. Ton rôle est de créer un plan d'action séquentiel clair pour le Codeur.
    
    Colonnes du dataset : {metadata['columns']}
    Types: {metadata['types']}
    
    Réponds DIRECTEMENT avec le plan (étapes 1, 2, 3...) et précise quel type de graphique (si pertinent) doit être généré.
    N'ÉCRIS SURTOUT PAS DE CODE PYTHON. UNIQUEMENT DES INSTRUCTIONS EN FRANÇAIS.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Conversion de la réponse texte en liste d'étapes (approximation simple)
    plan_text = response.content.strip()
    plan_steps = [line.strip() for line in plan_text.split('\n') if line.strip()]
    
    print("🎯 [Planner] Plan généré :", plan_steps)
    return {"current_plan": plan_steps}
