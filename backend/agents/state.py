from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Représente l'état global du graphe LangGraph à tout moment.
    """
    session_id: str
    user_prompt: str
    dataset_metadata: Dict[str, Any]
    
    messages: List[Dict[str, str]]
    
    # Workflow Multi-Agents Supervisor
    next_agent: Optional[str]
    active_agent: Optional[str]
    
    # Workflow Exécution
    current_plan: Optional[List[str]]
    generated_code: Optional[str]
    execution_result: Optional[str]
    plot_json: Optional[str]
    
    # Gestion d'erreurs (Self-Healing)
    error: Optional[str]
    retry_count: int
    
    final_answer: Optional[str]
