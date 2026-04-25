import json
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.config import settings
from backend.agents.state import AgentState
from backend.agents.prompts import DATA_SCIENTIST_PROMPT

llm = None
def get_llm():
    global llm
    if llm is None:
        llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, format="json")
    return llm

def data_analyst_agent(state: AgentState) -> dict:
    """
    Agent dédié à l'analyse de données zéro-shot / few-shot.
    Renvoie un JSON structuré tel que défini par le DATA_SCIENTIST_PROMPT.
    """
    print("📊 [DataAnalyst] Prise en charge de la requête...")
    llm_instance = get_llm()
    
    metadata = state["dataset_metadata"]
    prompt = state["user_prompt"]
    
    # Construction du message utilisateur exact comme demandé
    user_message = f"User Query:\n{prompt}\n\nDataset Metadata:\n{json.dumps(metadata, indent=2)}\n\nSample Data:\n{metadata.get('sample', [])}"
    
    messages = [
        SystemMessage(content=DATA_SCIENTIST_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    response = llm_instance.invoke(messages)
    
    try:
        # On s'assure que c'est bien formatté
        parsed = json.loads(response.content)
        # On le passe dans final_answer sous forme de json string pour l'instant
        # pour éviter de casser l'orchestrator qui attend execution_result ou final_answer string
        return {
            "execution_result": response.content,
            "active_agent": "data_analyst",
            "next_node": "end" # Remplacement direct, fini
        }
    except Exception as e:
        print(f"⚠️ [DataAnalyst] Erreur JSON: {e}")
        return {
            "execution_result": "Erreur lors de la génération de l'analyse json.",
            "active_agent": "data_analyst"
        }

def visualization_agent(state: AgentState) -> dict:
    """
    Agent dédié à la création de graphiques spécialisés avec Plotly.
    """
    print("📈 [Visualization] Prise en charge de la requête...")
    plan = [
        "1. Générer le graphique Plotly Express demandé.",
        "2. Exporter le json via fig.to_json() dans 'plot_json'.",
        "3. Assigner l'explication à 'result'."
    ]
    
    return {
        "current_plan": plan,
        "active_agent": "visualization",
        "next_node": "coder"
    }
