from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Représente l'état global du graphe LangGraph à tout moment.
    """
    session_id: str
    user_prompt: str
    dataset_metadata: Dict[str, Any]
    
    messages: List[Dict[str, str]]
    
    # Workflow Multi-Agents
    next_agent: Optional[str]
    active_agent: Optional[str]
    agent_trace: List[str] # Historique des agents activés
    
    # Workflow Exécution
    current_plan: Optional[List[str]]
    generated_code: Optional[str]
    execution_result: Optional[Any]
    plot_json: Optional[str]
    all_plots: List[str] # Stockage de plusieurs graphiques si nécessaire
    
    # Résultats accumulés (pour le Reporter)
    insights: List[str]
    anomalies: List[str]
    correlations: List[str]
    forecast_results: Dict[str, Any]
    critique: Dict[str, Any]
    
    # Gestion d'erreurs (Self-Healing)
    error: Optional[str]
    retry_count: int
    
    agent_thought: Optional[str]
    final_answer: Optional[str]
    structured_output: Optional[Dict[str, Any]]
