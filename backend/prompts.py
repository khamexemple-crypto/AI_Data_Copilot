"""
Optimized and compact prompts for the AI Data Copilot.
Reduces token usage and improves local LLM reliability and speed.
"""

from backend.performance import compress_metadata

# ──────────────────────────────────────────────
# PLANNER
# ──────────────────────────────────────────────

def build_planner_prompt(user_query: str, metadata: dict, sample_rows: list) -> tuple[str, str]:
    system = """You are Planner Agent. Classify the user request and return a short execution plan.
Task types: analysis, visualization, prediction, data_quality, general_question.
Return JSON ONLY. No markdown. No extra text.
{
  "task_type": "analysis",
  "steps": ["step 1"],
  "reasoning_summary": "Short reason."
}"""

    context = compress_metadata(metadata, sample_rows)
    user = f"User query: {user_query}\nDataset info:\n{context}"

    return system, user


# ──────────────────────────────────────────────
# ANALYST
# ──────────────────────────────────────────────

def build_analyst_prompt(user_query: str, metadata: dict, sample_rows: list, plan: dict) -> tuple[str, str]:
    system = """You are Analyst Agent. Extract insights, anomalies, correlations, and key columns based on data.
Keep answers short and factual.
Return JSON ONLY. No markdown. No extra text.
{
  "insights": ["insight 1"],
  "anomalies": ["anomaly 1"],
  "correlations": ["correlation 1"],
  "important_columns": ["col1"]
}"""

    context = compress_metadata(metadata, sample_rows)
    user = f"User query: {user_query}\nPlan: {plan}\nDataset info:\n{context}"

    return system, user


# ──────────────────────────────────────────────
# REVIEWER
# ──────────────────────────────────────────────

def build_reviewer_prompt(user_query: str, analyst_output: dict, metadata: dict) -> tuple[str, str]:
    system = """You are Reviewer Agent. Critique the Analyst output.
Find weaknesses, limitations, and score confidence (0.0 to 1.0).
Return JSON ONLY. No markdown. No extra text.
{
  "issues": ["issue 1"],
  "limitations": ["limitation 1"],
  "confidence": 0.8
}"""

    # We don't need full sample rows here, just metadata to save tokens
    context = compress_metadata(metadata, [])
    user = f"Query: {user_query}\nAnalyst output:\n{analyst_output}\nDataset shape/cols:\n{context}"

    return system, user


# ──────────────────────────────────────────────
# REPORTER
# ──────────────────────────────────────────────

def build_reporter_prompt(user_query: str, plan: dict, analysis: dict, critic: dict) -> tuple[str, str]:
    system = """You are Reporter Agent. Summarize the findings into a clear, short final answer.
Return JSON ONLY. No markdown. No extra text.
{
  "summary": "1-2 sentences summary.",
  "final_answer": "Detailed answer for the user."
}"""

    user = f"Query: {user_query}\nPlan: {plan}\nAnalysis: {analysis}\nCritic: {critic}"

    return system, user


# ──────────────────────────────────────────────
# FAST AGENT
# ──────────────────────────────────────────────

def build_fast_prompt(user_query: str, metadata: dict, sample_rows: list) -> tuple[str, str]:
    system = """You are Fast Analyst Agent. Perform a quick, single-pass analysis.
Return JSON ONLY matching the exact schema below. No markdown. No extra text.
{
  "plan": {
    "task_type": "fast_analysis",
    "steps": ["Fast analysis"],
    "reasoning_summary": "Fast mode execution."
  },
  "analysis": {
    "insights": ["insight 1"],
    "anomalies": ["anomaly 1"],
    "correlations": ["correlation 1"],
    "important_columns": ["col1"]
  },
  "critic": {
    "issues": ["Fast mode limitation"],
    "limitations": ["Fast mode provides a summarized analysis and may be less exhaustive."],
    "confidence": 0.6
  },
  "report": {
    "summary": "Short summary.",
    "final_answer": "Complete final answer."
  }
}"""

    context = compress_metadata(metadata, sample_rows)
    user = f"User query: {user_query}\nDataset info:\n{context}"

    return system, user
