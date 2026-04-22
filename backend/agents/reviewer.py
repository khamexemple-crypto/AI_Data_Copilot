from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.coder import get_llm
from backend.agents.state import AgentState

def reviewer_agent(state: AgentState) -> dict:
    """
    Le Reviewer Agent: Analyse l'erreur d'exécution et prépare un message correctif pour le Codeur.
    """
    llm = get_llm()
    code = state.get("generated_code")
    error = state.get("error")
    
    system_prompt = f"""
    Tu es un Senior Python Reviewer très technique.
    Le "Data Scientist" a écrit du code Pandas qui a foiré. 
    
    CODE:
    {code}
    
    TRACEBACK (ERREUR):
    {error}
    
    TACHE: Formuler une recommandation très brève (1 à 2 phrases) pour corriger ce code, en pointant précisément la source du problème (ex: colonne manquante, mauvaise indentation, etc.). Ne propose pas le code entier, juste le conseil technique.
    """
    
    messages = [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    
    review_feedback = response.content.strip()
    print("🔍 [Reviewer] Feedback :", review_feedback)
    
    # On écrase le message d'erreur brut avec une version plus "intelligente" et guidée
    augmented_error = f"Erreur de l'interpréteur:\n{error}\n\n--- Conseil du Reviewer ---\n{review_feedback}"
    
    return {"error": augmented_error}
