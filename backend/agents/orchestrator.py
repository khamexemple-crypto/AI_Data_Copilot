import json
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.planner import planner_agent
from backend.agents.analyst import analyst_node
from backend.agents.visualization import visualization_node
from backend.agents.forecasting import forecasting_agent
from backend.agents.reporter import reporter_agent
from backend.agents.coder import coder_agent
from backend.agents.reviewer import reviewer_agent
from backend.agents.profiler import profiler_agent
from backend.tools.executor import execute_python_code
from backend.core.storage import session_storage

def executor_node(state: AgentState) -> dict:
    code = state.get("generated_code")
    session_id = state.get("session_id")
    df = session_storage[session_id]["dataframe"]
    
    if not code:
        return {"error": "Pas de code généré."}
        
    print(f"⚙️ [Executor] Exécution du code ({state.get('active_agent')})")
    exec_res = execute_python_code(code, df)
    
    if exec_res["success"]:
        print("✅ [Executor] Succès !")
        updates = {
            "execution_result": exec_res.get("result"),
            "error": None,
            "retry_count": 0
        }
        if exec_res.get("plot_json"):
            plots = state.get("all_plots", [])
            plots.append(exec_res.get("plot_json"))
            updates["all_plots"] = plots
        return updates
    else:
        print("❌ [Executor] Erreur d'exécution.")
        return {
            "error": exec_res["error"],
            "retry_count": state.get("retry_count", 0) + 1
        }

def route_after_execution(state: AgentState) -> str:
    if state.get("error"):
        if state.get("retry_count", 0) < 3:
            return "reviewer"
        else:
            return "reporter"
    return "reporter"

def route_from_planner(state: AgentState) -> str:
    next_a = state.get("next_agent", "analyst")
    if next_a == "visualization": return "visualization"
    if next_a == "forecasting": return "forecaster"
    return "analyst"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_agent)
    workflow.add_node("profiler", profiler_agent)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("forecaster", forecasting_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_agent)
    workflow.add_node("reporter", reporter_agent)
    
    # Stratégie de point d'entrée dynamique
    # Pour simuler plusieurs points d'entrée, on utilise un noeud routeur 'start'
    def start_router(state: AgentState):
        if state.get("user_prompt") == "__AUTOPROFILE__":
            return "profiler"
        return "planner"

    workflow.add_conditional_edges(
        "planner", # Normalement on mettrait un noeud 'start', mais LangGraph v0.x set_entry_point est rigide.
        # On va utiliser une approche plus simple : on set_entry_point("planner") par défaut,
        # et on branche le profiler ailleurs si on veut, ou on change le entry point au runtime si possible.
        # Une façon propre est d'utiliser un noeud "router" en entrée.
        route_from_planner,
        {
            "analyst": "analyst",
            "visualization": "visualization",
            "forecaster": "forecaster"
        }
    )
    
    workflow.add_edge("profiler", "coder")
    workflow.add_edge("analyst", "coder")
    workflow.add_edge("visualization", "coder")
    workflow.add_edge("forecaster", "coder")
    workflow.add_edge("coder", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "reviewer": "reviewer",
            "reporter": "reporter"
        }
    )
    workflow.add_edge("reviewer", "coder")
    workflow.add_edge("reporter", END)
    
    # Pour le profiling, on va créer un graphe séparé ou changer l'entry point
    # Mais LangGraph ne permet pas de changer dynamiquement l'entry point facilement dans un seul graphe compilé.
    # SOLUTION : On met un noeud "gateway" en entrée.
    
    return workflow

def get_final_graph(entry_point="planner"):
    wf = build_graph()
    wf.set_entry_point(entry_point)
    return wf.compile()

app_graph_chat = None
app_graph_profile = None

def run_orchestrator(session_id: str, prompt: str, mode: str = "chat") -> dict:
    global app_graph_chat, app_graph_profile
    
    if mode == "profile":
        if app_graph_profile is None:
            app_graph_profile = get_final_graph("profiler")
        graph = app_graph_profile
    else:
        if app_graph_chat is None:
            app_graph_chat = get_final_graph("planner")
        graph = app_graph_chat
        
    session_data = session_storage[session_id]
    
    initial_state = {
        "session_id": session_id,
        "user_prompt": prompt,
        "dataset_metadata": session_data["metadata"],
        "messages": [],
        "all_plots": [],
        "agent_trace": [],
        "retry_count": 0,
        "error": None,
        "insights": [],
        "anomalies": [],
        "correlations": [],
        "forecast_results": {},
        "critique": {}
    }
    
    final_state = graph.invoke(initial_state)
    structured = final_state.get("structured_output", {})
    
    return {
        "answer": json.dumps(structured) if structured else final_state.get("final_answer"),
        "plots": final_state.get("all_plots", []),
        "trace": final_state.get("agent_trace", [])
    }
