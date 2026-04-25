from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.config import settings
from backend.agents.state import AgentState

# Initialisation du LLM en lazy (Singleton local)
llm = None

def get_llm():
    global llm
    if llm is None:
        llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)
    return llm

def coder_agent(state: AgentState) -> dict:
    """
    Le Coder Agent: Traduit la requête (ou le plan) en code Python Panda.
    """
    llm_instance = get_llm()
    prompt = state["user_prompt"]
    metadata = state["dataset_metadata"]
    error = state.get("error")
    
    system_prompt = f"""
    Tu es un Data Scientist expert en Python et structure Pandas.
    Le dataset disponible s'appelle `df` et est déjà chargé.
    Caractéristiques du dataset:
    - Colonnes : {metadata['columns']}
    - Types : {metadata['types']}
    - Lignes/Col : {metadata['shape']}
    - Sample : {metadata['sample']}
    
    TACHE : Satisfaire la demande en générant DU CODE PYTHON EXECUTABLE ET RIEN D'AUTRE.
    
    CONTRAINTES CRITIQUES :
    1. Stocke explicitement le résultat du calcul OU l'insight qualitatif dans une variable locale nommée `result` (obligatoirement de type string). SI TON RESULTAT EST UN DATAFRAME OU UNE SERIE PANDAS, uilises `.to_markdown()` au lieu de `.to_string()` pour qu'il s'affiche correctement sous forme de tableau. Prends soin de faire le nécessaire pour que l'affichage soit lisible.
    2. Si 'graph', 'plot' ou 'visualiser' est mentionné, tu DOIS générer un graphique plotly.express, et stocker l'objet json généré dans la variable `plot_json` grâce à `fig.to_json()`. Ne fais pas fig.show().
    3. Ton output doit UNIQUEMENT contenir du code Python pur. Pas de format Markdown (```python ... ```), pas d'explications textuelles.
    """
    
    if "current_plan" in state and state["current_plan"]:
        plan_str = "\n".join(state["current_plan"])
        system_prompt += f"\n\n🤖 PLAN GLOBAL :\n{plan_str}"
        
    if state.get("agent_thought"):
        system_prompt += f"\n\n🧠 RÉFLEXION DU SPÉCIALISTE ({state.get('active_agent')}) :\n{state.get('agent_thought')}"
        
    if error:
        system_prompt += f"\n\n[!! DANGER !!] La précédente exécution a généré une erreur : \n{error}\nVeuillez corriger le code !"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm_instance.invoke(messages)
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    trace = state.get("agent_trace", [])
    trace.append(f"Coder (Target: {state.get('active_agent')})")
    
    print(f"💻 [Coder] Code généré pour l'agent {state.get('active_agent')}")
    return {
        "generated_code": code,
        "agent_trace": trace
    }
