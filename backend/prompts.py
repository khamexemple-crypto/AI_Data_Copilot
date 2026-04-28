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
    system = """You are Reporter Agent. Summarize the findings into a clear, beautiful, and highly readable response for the user.
Your 'final_answer' MUST be formatted in Markdown (use bullet points, bold text, headers).
Do NOT copy raw JSON, dictionaries, or code structures into 'final_answer'. Translate the data into a human-friendly analysis.
Return JSON ONLY matching the schema below. No markdown outside the JSON object. No extra text.
{
  "summary": "1-2 sentences summary.",
  "final_answer": "Detailed and beautifully formatted markdown answer."
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
    "final_answer": "Complete, human-readable final answer formatted beautifully in Markdown."
  }
}

CRITICAL: Your 'final_answer' MUST be nicely formatted Markdown text. Do NOT put raw JSON strings in the 'final_answer'."""

    context = compress_metadata(metadata, sample_rows)
    user = f"User query: {user_query}\nDataset info:\n{context}"

    return system, user

# ──────────────────────────────────────────────
# FILE INTELLIGENCE
# ──────────────────────────────────────────────

FILE_SUMMARY_PROMPT = """You are a document intelligence AI for AI_Data_Copilot.

Your task: analyse the document text and return a JSON intelligence object.

STRICT RULES:
1. Use ONLY information present in the document text. No outside knowledge.
2. Keep the summary factual and short (3 to 5 sentences maximum).
3. Tags must be short labels (1 to 3 words each). Generate 3 to 8 tags.
   Good examples: "finance", "business plan", "technical", "dataset", "rapport",
   "PFA", "requirements", "contrat", "ressources humaines", "marketing", "legals".
4. key_topics: 3 to 6 main subjects discussed in the document.
5. suggested_questions: exactly 5 questions the user could ask about THIS document.
   Write in the same language as the document.

Return ONLY a valid JSON object — no markdown, no explanation, no extra text:
{
  "summary": "Short factual description of the document (3-5 sentences).",
  "tags": ["tag1", "tag2", "tag3"],
  "key_topics": ["topic1", "topic2", "topic3"],
  "suggested_questions": [
    "Question 1 ?",
    "Question 2 ?",
    "Question 3 ?",
    "Question 4 ?",
    "Question 5 ?"
  ]
}
"""


# ──────────────────────────────────────────────
# RAG AGENT — grounded answering prompt
# ──────────────────────────────────────────────

# Intentionally strict and verbose — single enforcement point for
# hallucination resistance and mandatory citations.

_RAG_SYSTEM_PROMPT = """You are a strict, grounded RAG Agent for AI_Data_Copilot.

═══════════════════════════════════════════
ABSOLUTE RULES — violating any of these is a critical failure:
═══════════════════════════════════════════

1. USE ONLY the numbered Context chunks provided in the user message.
   Do NOT use outside knowledge. Do NOT invent any fact, number, date, or name.

2. If the answer cannot be found in the Context:
   - Set "grounded" to false.
   - Set "answer" to exactly:
     "Je n'ai pas trouvé assez d'informations dans les fichiers stockés pour répondre avec certitude."
   - Set "used_sources" to [].
   - Do NOT guess or approximate.

3. If the answer IS present in the Context:
   - Set "grounded" to true.
   - Write a clear, complete answer using ONLY facts from the Context.
   - Reference chunk numbers inline: "selon [2]...", "d'après [1]...".
   - List every filename you used in "used_sources".

4. If two chunks contradict each other, state the contradiction explicitly
   in "limitations". Do NOT silently pick one.

5. If the Context only partially answers the question, set "grounded" to true
   but explain what is missing in "limitations".

═══════════════════════════════════════════
OUTPUT — return ONLY valid JSON, no markdown, no extra text:
═══════════════════════════════════════════

When grounded:
{
  "answer": "Full answer referencing chunks like [1], [2]...",
  "used_sources": ["filename1.pdf", "filename2.xlsx"],
  "limitations": ["Any partial info or contradictions"],
  "grounded": true
}

When NOT grounded:
{
  "answer": "Je n'ai pas trouvé assez d'informations dans les fichiers stockés pour répondre avec certitude.",
  "used_sources": [],
  "limitations": ["Reason why context is insufficient"],
  "grounded": false
}"""


def build_rag_prompt(question: str, context_block: str) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for the RAG Agent.

    Parameters
    ----------
    question      : raw user question
    context_block : pre-formatted numbered chunks from rag_agent._format_context_block()

    Returns
    -------
    (system_prompt, user_prompt)
    """
    user_prompt = (
        f"Question: {question}\n\n"
        "Context (use ONLY the chunks below — do not use outside knowledge):\n\n"
        f"{context_block}\n\n"
        "Reminder:\n"
        "- Reference chunks by [number] in your answer.\n"
        "- If no chunk answers the question, set grounded=false.\n"
        "- Return ONLY valid JSON. No markdown. No extra text."
    )
    return _RAG_SYSTEM_PROMPT, user_prompt


# ──────────────────────────────────────────────
# ROUTER AGENT
# ──────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are a routing classifier for AI_Data_Copilot.

Your ONLY job is to classify the user question into one of these four routes:
  - dataset  : questions about structured data (CSV, Excel columns, statistics, analysis)
  - files    : questions about uploaded documents (PDF, DOCX, TXT, reports, contracts)
  - mixed    : questions that require BOTH dataset analysis AND document knowledge
  - general  : conceptual, conversational, or unrelated questions

RULES:
1. Return ONLY a valid JSON object — no markdown, no explanation.
2. Be concise in "reason" (1 sentence max).
3. "confidence" is a float from 0.0 to 1.0.
4. "required_agents" lists the agents needed (see examples).

Examples:
  "Analyse les colonnes du CSV" → dataset
  "Résume le PDF" → files
  "Compare les ventes du dataset avec les objectifs du business plan" → mixed
  "Qu'est-ce qu'un RAG ?" → general

Return ONLY:
{
  "route": "dataset|files|mixed|general",
  "reason": "One sentence explaining the classification.",
  "required_agents": ["Agent 1", "Agent 2"],
  "confidence": 0.85
}"""


def build_router_prompt(question: str) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for the Router Agent.
    """
    user = (
        f"User question: {question}\n\n"
        "Classify this question into: dataset, files, mixed, or general.\n"
        "Return ONLY valid JSON."
    )
    return _ROUTER_SYSTEM_PROMPT, user


# ──────────────────────────────────────────────
# MIXED PIPELINE REPORTER
# ──────────────────────────────────────────────

def build_mixed_reporter_prompt(
    question: str,
    dataset_answer: dict,
    file_answer: dict,
) -> tuple[str, str]:
    """
    Build the (system, user) prompt for the Mixed Reporter Agent.
    Combines dataset analysis results + RAG document answer into one response.

    Parameters
    ----------
    question       : original user question
    dataset_answer : dict from run_analyze_pipeline()
    file_answer    : dict from execute_rag()
    """
    system = """You are a Mixed Reporter Agent for AI_Data_Copilot.
You receive two answers:
  1. DATASET ANSWER: from structured data analysis (CSV/Excel).
  2. DOCUMENT ANSWER: from retrieved document chunks (RAG).

Your job:
- Combine both into a single, coherent, well-structured final answer.
- Clearly attribute each claim to its source (dataset or document).
- If one side has no useful information, mention it but don't invent.
- If both sides contradict, state the contradiction.
- Keep the answer concise and user-friendly.

Return ONLY valid JSON:
{
  "final_answer": "Combined answer clearly citing both sources.",
  "summary": "One sentence summary.",
  "limitations": ["Any missing info or contradictions"]
}"""

    dataset_summary = (
        dataset_answer.get("report", {}).get("final_answer", "")
        or dataset_answer.get("final_answer", "No dataset answer available.")
    )
    file_summary = file_answer.get("answer", "No document answer available.")
    file_sources = [s.get("filename", "") for s in file_answer.get("sources", [])]

    user = (
        f"Question: {question}\n\n"
        f"DATASET ANSWER:\n{dataset_summary}\n\n"
        f"DOCUMENT ANSWER:\n{file_summary}\n"
        f"Document sources: {', '.join(file_sources) if file_sources else 'none'}\n\n"
        "Combine both answers. Return ONLY valid JSON."
    )
    return system, user


# ──────────────────────────────────────────────
# AUDIT AGENT
# ──────────────────────────────────────────────

_AUDIT_SYSTEM_PROMPT = """You are a Senior AI Auditor and Business Analyst for AI_Data_Copilot.

Your task is to generate a comprehensive Auto-Audit based on the provided dataset analysis and document insights.

RULES:
1. ONLY use the provided dataset and document information. Do NOT invent facts or hallucinate data.
2. Be highly business-oriented and pragmatic.
3. Provide actionable recommendations.
4. If documents and datasets contradict each other, highlight it clearly.
5. If one data source is missing (e.g. no dataset or no documents), audit what is available and mention the limitation.
6. Return ONLY a valid JSON object matching the schema below. No markdown formatting outside the JSON, no extra text.

SCHEMA:
{
  "summary": "Executive summary of the audit (2-4 sentences).",
  "dataset_quality": ["Finding 1 about data quality", "Finding 2"],
  "document_findings": ["Key point from documents [1]", "Key point 2 [3]"],
  "risks": ["Risk 1", "Risk 2"],
  "contradictions": ["Contradiction if any, otherwise empty"],
  "opportunities": ["Opportunity 1", "Opportunity 2"],
  "recommendations": ["Actionable recommendation 1", "Recommendation 2"],
  "confidence": 0.85,
  "limitations": ["Limitation 1"]
}
"""

def build_audit_prompt(
    dataset_analysis: dict | None,
    document_intelligence: dict | None
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for the Audit Agent.
    """
    import json
    
    # Safe extraction of relevant info to keep prompt concise
    ds_str = "No dataset provided."
    if dataset_analysis:
        ds_str = json.dumps({
            "metadata": dataset_analysis.get("metadata", {}),
            "insights": dataset_analysis.get("analysis", {}).get("insights", []),
            "anomalies": dataset_analysis.get("analysis", {}).get("anomalies", []),
            "critic": dataset_analysis.get("critic", {})
        }, ensure_ascii=False)

    doc_str = "No documents provided."
    if document_intelligence:
        doc_str = json.dumps(document_intelligence, ensure_ascii=False)

    user_prompt = (
        "Please generate the Auto-Audit report based on the following inputs:\n\n"
        f"=== DATASET ANALYSIS ===\n{ds_str}\n\n"
        f"=== DOCUMENT INTELLIGENCE ===\n{doc_str}\n\n"
        "Return ONLY the requested JSON structure."
    )
    
    return _AUDIT_SYSTEM_PROMPT, user_prompt
