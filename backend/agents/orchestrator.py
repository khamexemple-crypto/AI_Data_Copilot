from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.supervisor import supervisor_agent
from backend.agents.specialized_agents import data_analyst_agent, visualization_agent
from backend.agents.planner import planner_agent
from backend.agents.coder import coder_agent
from backend.agents.reviewer import reviewer_agent
from backend.tools.executor import execute_python_code
from backend.core.storage import session_storage

def executor_node(state: AgentState) -> dict:
    code = state.get("generated_code")
    session_id = state.get("session_id")
    df = session_storage[session_id]["dataframe"]
    
    if not code:
        return {"error": "Pas de code généré par l'agent Coder."}
        
    print("⚙️ [Executor] Exécution du code généré...")
    # Exécution sécurisée
    exec_res = execute_python_code(code, df)
    
    if exec_res["success"]:
        print("✅ [Executor] Succès !")
        return {
            "execution_result": str(exec_res.get("result", "")),
            "plot_json": exec_res.get("plot_json"),
            "error": None,
            "final_answer": "Exécution réussie."
        }
    else:
        print("❌ [Executor] Erreur d'exécution.")
        # En cas d'erreur, on incrémente le retry_count
        return {
            "error": exec_res["error"],
            "retry_count": state.get("retry_count", 0) + 1
        }

def route_after_execution(state: AgentState) -> str:
    """
    Détermine si l'on continue ou s'il y a eu une erreur rattrapable.
    """
    if state.get("error"):
        if state.get("retry_count", 0) < 3:
            print(f"🔄 [Router] Redirection vers le Reviewer (Essai {state.get('retry_count')}/3)")
            return "reviewer"
        else:
            print("🚨 [Router] Echec critique après 3 tentatives.")
            return "end"
    return "end"

def route_from_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "planner")
    if next_agent not in ["planner", "data_analyst", "visualization"]:
        print(f"⚠️ [Router] Agent inconnu '{next_agent}', fallback -> planner")
        return "planner"
    return next_agent

def build_graph():
    """
    Construit le Graphe d'état Multi-Agents avec Supervisor et agents spécialisés.
    """
    workflow = StateGraph(AgentState)
    
    # Noeuds
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("data_analyst", data_analyst_agent)
    workflow.add_node("visualization", visualization_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_agent)
    
    # Workflow logique
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "data_analyst": "data_analyst",
            "visualization": "visualization"
        }
    )
    
    workflow.add_edge("planner", "coder")
    workflow.add_edge("data_analyst", "coder")
    workflow.add_edge("visualization", "coder")
    workflow.add_edge("coder", "executor")
    
    # Self-Healing Loop (Error Path)
    workflow.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "reviewer": "reviewer",
            "end": END
        }
    )
    workflow.add_edge("reviewer", "coder") # Le reviewer passe le relais au codeur
    
    return workflow.compile()

app_graph = None

def run_orchestrator(session_id: str, prompt: str) -> dict:
    """
    Point d'entrée de notre Backend Agentique.
    """
    global app_graph
    if app_graph is None:
        app_graph = build_graph()
        
    session_data = session_storage[session_id]
    
    initial_state = {
        "session_id": session_id,
        "user_prompt": prompt,
        "dataset_metadata": session_data["metadata"],
        "messages": [],
        "retry_count": 0,
        "error": None
    }
    
    print(f"🚀 [Orchestrator] Démarrage du flow pour la session {session_id} avec prompt: '{prompt}'")
    final_state = app_graph.invoke(initial_state)
    print(f"🏁 [Orchestrator] Fin du flow.")
    
    # Extraction propre des résultats pour l'UI
    answer = final_state.get("execution_result") 
    if not answer:
        answer = final_state.get("error", "Pas de résultat clair.")
        if final_state.get("error") and final_state.get("retry_count", 0) >= 3:
            answer = f"🚨 L'agent a échoué à écrire un code fonctionnel après 3 tentatives.\n**Erreur technique:**\n{final_state.get('error')}"
            
    plots = []
    if final_state.get("plot_json"):
        plots.append(final_state.get("plot_json"))
        
    return {
        "answer": answer,
        "plots": plots
    }
