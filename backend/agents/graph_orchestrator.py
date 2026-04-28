import json
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.config import settings

# ──────────────────────────────────────────────
# 1. STATE DEFINITION
# ──────────────────────────────────────────────
class OrchestratorState(TypedDict):
    user_query: str
    plan: List[Dict[str, Any]]
    current_step: int
    data: Any # DataFrame 
    intermediate_results: Dict[str, Any]
    final_output: Any
    critic_feedback: str
    critic_approved: bool
    retry_count: int
    current_agent: str

def get_llm(json_format=False):
    fmt = "json" if json_format else None
    return ChatOllama(model=settings.DEFAULT_MODEL, base_url=settings.OLLAMA_BASE_URL, format=fmt)

# ──────────────────────────────────────────────
# 2. ORCHESTRATOR NODE
# ──────────────────────────────────────────────
def orchestrator_node(state: OrchestratorState) -> Dict:
    print("🧠 [Orchestrator] Planning steps...")
    llm = get_llm(json_format=True)
    query = state.get("user_query", "")
    feedback = state.get("critic_feedback", "")
    
    system_prompt = """
    You are the central Orchestrator for an AI Data Copilot.
    Your job is to break down the user's request into a multi-step plan.
    Available agents:
    - DataAnalystAgent: Performs Exploratory Data Analysis (EDA), summary stats.
    - DataCleaningAgent: Handles missing values, outliers.
    - MLAgent: Trains predictive models (classification/regression) and returns metrics.
    - VisualizationAgent: Generates plots.
    - SQLAgent: Translates natural language to SQL.
    
    You MUST return a JSON object with a 'plan' key containing a list of steps.
    Each step must have:
    - 'agent': the exact name of the agent to call
    - 'task': specific instructions for this step
    
    Example:
    {
      "plan": [
        {"agent": "DataAnalystAgent", "task": "Analyze the distribution of columns"},
        {"agent": "MLAgent", "task": "Train a random forest classifier"}
      ]
    }
    """
    
    user_msg = f"User query: {query}"
    if feedback:
         user_msg += f"\n\nPrevious attempt failed with feedback: {feedback}\nPlease adjust the plan if necessary."
         
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    
    try:
        result = json.loads(response.content)
        plan = result.get("plan", [])
    except Exception:
        # Fallback plan
        plan = [{"agent": "DataAnalystAgent", "task": query}]
        
    return {
        "plan": plan,
        "current_step": 0,
        "retry_count": 0,
        "critic_feedback": "",
        "critic_approved": False
    }

# ──────────────────────────────────────────────
# 3. AGENT NODES
# ──────────────────────────────────────────────
def data_analyst_node(state: OrchestratorState) -> Dict:
    print("📊 [DataAnalystAgent] Executing task...")
    llm = get_llm()
    step_info = state["plan"][state["current_step"]]
    task = step_info["task"]
    data = state.get("data")
    columns = list(data.columns) if data is not None else []
    
    sys_prompt = f"""You are the DataAnalystAgent.
    Your task: {task}
    Dataset columns: {columns}
    Return your analysis directly."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)])
    
    interm = state.get("intermediate_results", {})
    interm["DataAnalystAgent"] = response.content
    
    return {
        "intermediate_results": interm,
        "current_agent": "DataAnalystAgent"
    }

def ml_agent_node(state: OrchestratorState) -> Dict:
    print("🤖 [MLAgent] Executing task...")
    llm = get_llm()
    step_info = state["plan"][state["current_step"]]
    task = step_info["task"]
    data = state.get("data")
    columns = list(data.columns) if data is not None else []
    
    sys_prompt = f"""You are the MLAgent.
    Your task: {task}
    Dataset columns: {columns}
    Please output the pseudo-code or ML approach you would take, and expected metrics."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)])
    
    interm = state.get("intermediate_results", {})
    interm["MLAgent"] = response.content
    
    return {
        "intermediate_results": interm,
        "current_agent": "MLAgent"
    }

def stub_agent_node(state: OrchestratorState) -> Dict:
    print("🔧 [StubAgent] Executing stub task...")
    step_info = state["plan"][state["current_step"]]
    agent_name = step_info["agent"]
    
    interm = state.get("intermediate_results", {})
    interm[agent_name] = f"Stub execution for {agent_name}: Task completed."
    
    return {
        "intermediate_results": interm,
        "current_agent": agent_name
    }

# ──────────────────────────────────────────────
# 4. CRITIC AGENT
# ──────────────────────────────────────────────
def critic_node(state: OrchestratorState) -> Dict:
    print("🧐 [CriticAgent] Validating agent output...")
    llm = get_llm(json_format=True)
    agent_name = state.get("current_agent", "Unknown")
    
    # Safely get the output for the current agent
    interm = state.get("intermediate_results", {})
    output_to_evaluate = interm.get(agent_name, "")
    
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    task = plan[current_step]["task"] if current_step < len(plan) else "Unknown"
    
    sys_prompt = """
    You are the CriticAgent.
    Validate the output of the previous agent for:
    1. Correctness
    2. Logical consistency
    3. Hallucination risks
    
    Return a JSON object with:
    - 'approved': boolean (true if the output is acceptable, false otherwise)
    - 'feedback': string (explanation of why it was approved or rejected)
    """
    
    user_msg = f"Task: {task}\nAgent: {agent_name}\nOutput:\n{output_to_evaluate}"
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
    
    try:
        result = json.loads(response.content)
        approved = result.get("approved", True)
        feedback = result.get("feedback", "")
    except Exception:
        approved = True
        feedback = "Failed to parse critic feedback, defaulting to approved."
        
    print(f"🧐 [CriticAgent] Decision: {'Approved' if approved else 'Rejected'}")
        
    return {
        "critic_approved": approved,
        "critic_feedback": feedback
    }

def advance_step_node(state: OrchestratorState) -> Dict:
    print("✅ [Advance] Moving to next step...")
    return {
        "current_step": state.get("current_step", 0) + 1,
        "retry_count": 0,
        "critic_feedback": "",
        "critic_approved": False
    }

def reject_step_node(state: OrchestratorState) -> Dict:
    print("❌ [Reject] Retrying step or replanning...")
    return {
        "retry_count": state.get("retry_count", 0) + 1
    }

# ──────────────────────────────────────────────
# 5. GRAPH ROUTING LOGIC
# ──────────────────────────────────────────────
def route_to_agent(state: OrchestratorState) -> str:
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if not plan or current_step >= len(plan):
        return END
        
    next_agent = plan[current_step].get("agent", "")
    
    if next_agent == "DataAnalystAgent":
        return "DataAnalystAgent"
    elif next_agent == "MLAgent":
        return "MLAgent"
    else:
        return "StubAgent"

def route_after_critic(state: OrchestratorState) -> str:
    if state.get("critic_approved", True):
        return "AdvanceStep"
    else:
        return "RejectStep"

def route_after_reject(state: OrchestratorState) -> str:
    if state.get("retry_count", 0) >= 2:
        print("🔄 [Router] Max retries reached, going back to Orchestrator to replan.")
        return "Orchestrator"
    else:
        # Retry the current agent
        print(f"🔄 [Router] Retrying current agent: {state.get('current_agent')}")
        current_agent = state.get("current_agent", "")
        if current_agent in ["DataAnalystAgent", "MLAgent"]:
            return current_agent
        return "StubAgent"

def build_graph() -> StateGraph:
    workflow = StateGraph(OrchestratorState)
    
    # Add Nodes
    workflow.add_node("Orchestrator", orchestrator_node)
    workflow.add_node("DataAnalystAgent", data_analyst_node)
    workflow.add_node("MLAgent", ml_agent_node)
    workflow.add_node("StubAgent", stub_agent_node)
    workflow.add_node("CriticAgent", critic_node)
    workflow.add_node("AdvanceStep", advance_step_node)
    workflow.add_node("RejectStep", reject_step_node)
    
    # Add Edges
    workflow.set_entry_point("Orchestrator")
    
    workflow.add_conditional_edges(
        "Orchestrator",
        route_to_agent,
        {
            "DataAnalystAgent": "DataAnalystAgent",
            "MLAgent": "MLAgent",
            "StubAgent": "StubAgent",
            END: END
        }
    )
    
    workflow.add_edge("DataAnalystAgent", "CriticAgent")
    workflow.add_edge("MLAgent", "CriticAgent")
    workflow.add_edge("StubAgent", "CriticAgent")
    
    workflow.add_conditional_edges(
        "CriticAgent",
        route_after_critic,
        {
            "AdvanceStep": "AdvanceStep",
            "RejectStep": "RejectStep"
        }
    )
    
    workflow.add_conditional_edges(
        "AdvanceStep",
        route_to_agent,
        {
            "DataAnalystAgent": "DataAnalystAgent",
            "MLAgent": "MLAgent",
            "StubAgent": "StubAgent",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "RejectStep",
        route_after_reject,
        {
            "Orchestrator": "Orchestrator",
            "DataAnalystAgent": "DataAnalystAgent",
            "MLAgent": "MLAgent",
            "StubAgent": "StubAgent"
        }
    )
    
    return workflow.compile()

# Example Execution
def run_multi_agent_workflow(query: str, df: Any = None) -> Dict:
    graph = build_graph()
    
    initial_state = {
        "user_query": query,
        "plan": [],
        "current_step": 0,
        "data": df,
        "intermediate_results": {},
        "final_output": None,
        "critic_feedback": "",
        "critic_approved": False,
        "retry_count": 0,
        "current_agent": ""
    }
    
    print(f"🚀 Starting Multi-Agent Orchestrator for query: {query}")
    final_state = graph.invoke(initial_state)
    
    # Format the final output
    final_state["final_output"] = final_state.get("intermediate_results", {})
    return final_state

if __name__ == "__main__":
    import pandas as pd
    
    # Minimal example dataframe
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 70000]
    })
    
    result = run_multi_agent_workflow("Analyze the dataset and build a prediction model for income", df)
    
    print("\n✅ Final Result:")
    print(json.dumps(result["final_output"], indent=2))
