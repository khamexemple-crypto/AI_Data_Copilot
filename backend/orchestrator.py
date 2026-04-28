"""
Orchestrateur séquentiel — pipeline : Planner → Analyst → Reviewer → Reporter.

Design :
- Pas de LangGraph : orchestration directe et lisible.
- Chaque agent est isolé dans son module.
- L'orchestrateur ne crashe jamais : il retourne "partial_success" si un agent échoue.
- Le format de sortie est stable et toujours valide.
"""

import time
from typing import Any

from backend.agents.planner import run_planner
from backend.agents.analyst import run_analyst
from backend.agents.reviewer import run_reviewer
from backend.agents.reporter import run_reporter
from backend.agents.fast_agent import run_fast_agent
from backend.core.storage import session_storage
from backend.performance import time_it
from backend.cache import generate_cache_key, get_cached_response, save_cached_response
from backend.fallbacks import (
    PLANNER_FALLBACK, ANALYST_FALLBACK, REVIEWER_FALLBACK, 
    REPORTER_FALLBACK, FAST_FALLBACK
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _run_agent(name: str, fallback: dict, fn, *args, **kwargs) -> tuple[dict, str, str | None, float]:
    """
    Exécute un agent de manière sécurisée et retourne le résultat, le statut, l'erreur, et le temps d'exécution.
    En cas d'erreur ou de timeout, retourne le fallback.
    """
    try:
        result, elapsed = time_it(fn, *args, **kwargs)
        return result, "success", None, elapsed
    except Exception as e:
        error_msg = f"{name} a planté ou timeout : {str(e)}"
        print(f"💥 [{name}] ERREUR CRITIQUE : {e}")
        return dict(fallback), "error", error_msg, 0.0


def _build_trace_entry(agent: str, status: str, output: dict) -> dict:
    return {"agent": agent, "status": status, "output": output}


# ──────────────────────────────────────────────
# Orchestrateur principal — /analyze
# ──────────────────────────────────────────────

def run_analyze_pipeline(
    session_id: str,
    user_query: str,
    mode: str = "deep",
    model_name: str | None = None,
    metadata: dict | None = None,
    sample_rows: list | None = None
) -> dict:
    """
    Pipeline complet : Planner → Analyst → Reviewer → Reporter.

    Paramètres :
        session_id  : ID de session (pour récupérer le DataFrame en mémoire)
        user_query  : question/requête de l'utilisateur
        metadata    : dict de métadonnées (optionnel, extrait de la session si absent)
        sample_rows : liste de dicts (lignes exemple) (optionnel)

    Retourne toujours un dict avec status "success" ou "partial_success".
    """
    start_time = time.time()
    warnings = []
    agent_trace = []

    # ── Récupération des données depuis la session ──────────────────────────
    if session_id in session_storage:
        session_data = session_storage[session_id]
        if metadata is None:
            metadata = session_data.get("metadata", {})
        if sample_rows is None:
            df = session_data.get("dataframe")
            if df is not None:
                sample_rows = df.head(10).to_dict(orient="records")
            else:
                sample_rows = []
    else:
        metadata = metadata or {}
        sample_rows = sample_rows or []
        warnings.append(f"Session '{session_id}' introuvable en mémoire — utilisation des données fournies directement.")

    # ── Résolution du Modèle ──────────────────────────────────────────────────
    from backend.core.config import settings
    if not model_name or model_name == "Auto recommended":
        resolved_model = settings.get_recommended_model(mode)
    else:
        resolved_model = model_name

    # ── Vérification du Cache ────────────────────────────────────────────────
    cache_key = generate_cache_key(user_query, mode, metadata, sample_rows)
    # The cache key currently doesn't include the model, which is fine if we consider 
    # the JSON output to be model-independent for caching purposes, but let's just keep it as is.
    cached_response = get_cached_response(cache_key)
    
    if cached_response:
        elapsed = round(time.time() - start_time, 2)
        print(f"📦 [Cache] Réponse servie depuis le cache en {elapsed}s")
        # Update performance metrics for cache hit
        cached_response["performance"]["total_time_sec"] = elapsed
        cached_response["performance"]["cache_hit"] = True
        return cached_response

    # ── Initialisation des temps d'agents ─────────────────────────────────────
    agent_times = {
        "fast_agent": 0.0,
        "planner": 0.0,
        "analyst": 0.0,
        "reviewer": 0.0,
        "reporter": 0.0
    }

    # ── Exécution Fast Mode ───────────────────────────────────────────────────
    if mode == "fast":
        fast_result, f_status, f_err, f_time = _run_agent(
            "Fast Agent", FAST_FALLBACK, run_fast_agent,
            user_query, metadata, sample_rows, resolved_model
        )
        agent_times["fast_agent"] = f_time
        agent_trace.append(_build_trace_entry("Fast Agent", f_status, fast_result))
        if f_err: warnings.append(f_err)
        
        elapsed = round(time.time() - start_time, 2)
        print(f"⚡ [Orchestrateur] Fast Mode terminé en {elapsed}s — statut : {f_status}")
        
        fallback_used = f_status == "error"
        
        response = fast_result.copy()
        response["status"] = "partial_success" if fallback_used else "success"
        response["mode"] = "fast"
        response["model"] = resolved_model
        response["agent_trace"] = agent_trace
        response["performance"] = {
            "total_time_sec": elapsed,
            "llm_time_sec": round(sum(agent_times.values()), 2),
            "agent_times": agent_times,
            "number_of_agents_used": 1,
            "cache_hit": False,
            "fallback_used": fallback_used
        }
        if warnings:
            response["warnings"] = warnings
            
        # Sauvegarde en cache
        save_cached_response(cache_key, response)
        return response

    # ── Exécution Deep Mode (par défaut) ──────────────────────────────────────

    # ── 1. Planner ──────────────────────────────────────────────────────────
    plan, p_status, p_err, p_time = _run_agent(
        "Planner Agent", PLANNER_FALLBACK, run_planner,
        user_query, metadata, sample_rows, resolved_model
    )
    agent_times["planner"] = p_time
    agent_trace.append(_build_trace_entry("Planner Agent", p_status, plan))
    if p_err:
        warnings.append(p_err)

    # ── 2. Analyst ──────────────────────────────────────────────────────────
    analysis, a_status, a_err, a_time = _run_agent(
        "Analyst Agent", ANALYST_FALLBACK, run_analyst,
        user_query, metadata, sample_rows, plan, resolved_model
    )
    agent_times["analyst"] = a_time
    agent_trace.append(_build_trace_entry("Analyst Agent", a_status, analysis))
    if a_err:
        warnings.append(a_err)

    # ── 3. Reviewer ─────────────────────────────────────────────────────────
    critic, r_status, r_err, r_time = _run_agent(
        "Reviewer Agent", REVIEWER_FALLBACK, run_reviewer,
        user_query, analysis, metadata, resolved_model
    )
    agent_times["reviewer"] = r_time
    agent_trace.append(_build_trace_entry("Reviewer Agent", r_status, critic))
    if r_err:
        warnings.append(r_err)

    # ── 4. Reporter ─────────────────────────────────────────────────────────
    report, rp_status, rp_err, rp_time = _run_agent(
        "Reporter Agent", REPORTER_FALLBACK, run_reporter,
        user_query, plan, analysis, critic, resolved_model
    )
    agent_times["reporter"] = rp_time
    agent_trace.append(_build_trace_entry("Reporter Agent", rp_status, report))
    if rp_err:
        warnings.append(rp_err)

    # ── Statut global ───────────────────────────────────────────────────────
    all_ok = all(
        entry["status"] == "success"
        for entry in agent_trace
    )
    global_status = "success" if all_ok else "partial_success"

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ [Orchestrateur] Pipeline terminé en {elapsed}s — statut : {global_status}")

    # ── Réponse finale ──────────────────────────────────────────────────────
    response = {
        "status": global_status,
        "mode": "deep",
        "model": resolved_model,
        "plan": {
            "task_type": plan.get("task_type", "analysis"),
            "steps": plan.get("steps", []),
            "reasoning_summary": plan.get("reasoning_summary", "")
        },
        "analysis": {
            "insights": analysis.get("insights", []),
            "anomalies": analysis.get("anomalies", []),
            "correlations": analysis.get("correlations", []),
            "important_columns": analysis.get("important_columns", [])
        },
        "critic": {
            "issues": critic.get("issues", []),
            "limitations": critic.get("limitations", []),
            "confidence": critic.get("confidence", 0.5)
        },
        "report": {
            "summary": report.get("summary", ""),
            "final_answer": report.get("final_answer", "")
        },
        "agent_trace": agent_trace,
        "performance": {
            "total_time_sec": elapsed,
            "llm_time_sec": round(sum(agent_times.values()), 2),
            "agent_times": agent_times,
            "number_of_agents_used": len(agent_trace),
            "cache_hit": False,
            "fallback_used": not all_ok
        }
    }

    if warnings:
        response["warnings"] = warnings

    # Sauvegarde en cache
    save_cached_response(cache_key, response)

    return response


# ──────────────────────────────────────────────
# Orchestrateur hérité — /chat et /profile (compatibilité)
# ──────────────────────────────────────────────

def run_orchestrator(session_id: str, prompt: str, mode: str = "chat") -> dict:
    """
    Point d'entrée pour les endpoints /chat et /profile existants.
    Redirige vers le pipeline principal et reformate la réponse
    au format attendu par le frontend Streamlit existant.
    """
    result = run_analyze_pipeline(
        session_id=session_id,
        user_query=prompt,
        mode="deep"
    )

    # Reformatage pour la compatibilité avec le frontend existant
    # (le frontend lit result["answer"] et result["plots"])
    structured = {
        "plan": result["plan"],
        "analysis": result["analysis"],
        "critic": result["critic"],
        "final_answer": result["report"].get("final_answer", ""),
        "summary": result["report"].get("summary", "")
    }

    return {
        "answer": __import__("json").dumps(structured, ensure_ascii=False),
        "plots": [],
        "trace": [entry["agent"] for entry in result["agent_trace"]],
        "status": result["status"]
    }


# ──────────────────────────────────────────────
# Unified /ask  — routes dataset / files / mixed / general
# ──────────────────────────────────────────────

def run_unified_ask(
    question   : str,
    session_id : str | None     = None,
    file_ids   : list[str]      = None,
    mode       : str            = "deep",
    model_name : str | None     = None,
    metadata   : dict | None    = None,
    sample_rows: list | None    = None,
) -> dict:
    """
    Single entry point that:
      1. Runs the Router Agent to classify the question.
      2. Dispatches to the correct pipeline(s).
      3. Returns a unified response envelope.

    Response schema
    ───────────────
    {
        "status"        : "success" | "partial_success" | "no_context",
        "route"         : "dataset" | "files" | "mixed" | "general",
        "router"        : { route, reason, required_agents, confidence, method },
        "dataset_answer": { ... }  or None,
        "file_answer"   : { ... }  or None,
        "final_answer"  : "string",
        "sources"       : [ ... ],
        "limitations"   : [ ... ],
        "agent_trace"   : [ ... ],
    }
    """
    import time
    start = time.time()
    file_ids = file_ids or []

    from backend.agents.router import route_question
    from backend.rag.rag_pipeline import execute_rag
    from backend.storage.file_registry import get_all_files
    from backend.llm import call_llm, safe_json_parse
    from backend.prompts import build_mixed_reporter_prompt

    # ── 1. Determine availability ────────────────────────────────────────────
    has_session = bool(session_id and session_id in session_storage)
    has_files   = bool(get_all_files())   # any indexed document in the registry

    # ── 2. Route ─────────────────────────────────────────────────────────────
    router_result = route_question(
        question,
        has_session = has_session,
        has_files   = has_files,
        model_name  = model_name,
    )
    route       = router_result["route"]
    agent_trace = [{"agent": "Router Agent", "status": "success", "output": router_result}]

    dataset_answer : dict | None = None
    file_answer    : dict | None = None
    final_answer   : str         = ""
    sources        : list        = []
    limitations    : list        = []
    status         = "success"

    # ── 3. Dispatch ───────────────────────────────────────────────────────────

    # ── dataset ──────────────────────────────────────────────────────────────
    if route in ("dataset", "mixed") and has_session:
        try:
            dataset_answer = run_analyze_pipeline(
                session_id  = session_id,
                user_query  = question,
                mode        = mode,
                model_name  = model_name,
                metadata    = metadata,
                sample_rows = sample_rows,
            )
            agent_trace.extend(dataset_answer.get("agent_trace", []))
            if dataset_answer.get("status") != "success":
                status = "partial_success"
        except Exception as e:
            limitations.append(f"Dataset pipeline error: {e}")
            status = "partial_success"

    # ── files ─────────────────────────────────────────────────────────────────
    if route in ("files", "mixed") and has_files:
        try:
            file_answer = execute_rag(question, model_name)
            agent_trace.append({
                "agent" : "RAG Agent",
                "status": "success" if file_answer.get("grounded") else "no_context",
                "output": {"confidence": file_answer.get("confidence", 0.0)},
            })
            sources     = file_answer.get("sources", [])
            limitations.extend(file_answer.get("limitations", []))
            if not file_answer.get("grounded"):
                status = "partial_success"
        except Exception as e:
            limitations.append(f"RAG pipeline error: {e}")
            status = "partial_success"

    # ── 4. Build final_answer ─────────────────────────────────────────────────

    if route == "dataset" and dataset_answer:
        report = dataset_answer.get("report", {})
        final_answer = report.get("final_answer", "") or dataset_answer.get("final_answer", "")

    elif route == "files" and file_answer:
        final_answer = file_answer.get("answer", "")
        if not file_answer.get("grounded"):
            status = "no_context"

    elif route == "mixed" and (dataset_answer or file_answer):
        # Mixed Reporter combines both sides
        system, user = build_mixed_reporter_prompt(
            question,
            dataset_answer or {},
            file_answer    or {},
        )
        try:
            from backend.core.config import settings
            resolved = model_name or settings.get_recommended_model(mode)
            raw    = call_llm(prompt=user, system=system, timeout=90, model_name=resolved)
            parsed = safe_json_parse(raw)
            if parsed and "final_answer" in parsed:
                final_answer = parsed["final_answer"]
                limitations.extend(parsed.get("limitations", []))
                agent_trace.append({
                    "agent" : "Mixed Reporter Agent",
                    "status": "success",
                    "output": {"summary": parsed.get("summary", "")},
                })
            else:
                # Fallback: concatenate both sides
                da = (dataset_answer or {}).get("report", {}).get("final_answer", "")
                fa = (file_answer or {}).get("answer", "")
                final_answer = f"**Analyse dataset:**\n{da}\n\n**Documents:**\n{fa}"
        except Exception as e:
            limitations.append(f"Mixed Reporter error: {e}")
            da = (dataset_answer or {}).get("report", {}).get("final_answer", "")
            fa = (file_answer or {}).get("answer", "")
            final_answer = f"{da}\n\n{fa}".strip()

    else:
        # General route or no data available
        final_answer = (
            "Je peux vous aider avec l'analyse de données et de documents. "
            "Uploadez un fichier CSV/Excel ou des documents PDF pour commencer."
        )
        status = "success"

    elapsed = round(time.time() - start, 2)

    return {
        "status"        : status,
        "route"         : route,
        "router"        : router_result,
        "dataset_answer": dataset_answer,
        "file_answer"   : file_answer,
        "final_answer"  : final_answer,
        "sources"       : sources,
        "limitations"   : limitations,
        "agent_trace"   : agent_trace,
        "performance"   : {"total_time_sec": elapsed},
    }


# ──────────────────────────────────────────────
# Auto-Audit Pipeline
# ──────────────────────────────────────────────

def run_auto_audit(
    session_id: str | None = None,
    file_ids: list[str] = None,
    mode: str = "deep",
    model_name: str | None = None,
    metadata: dict | None = None,
    sample_rows: list | None = None
) -> dict:
    import time
    start = time.time()
    file_ids = file_ids or []

    from backend.agents.audit_agent import run_audit_agent
    from backend.rag.rag_pipeline import execute_rag
    from backend.storage.file_registry import get_all_files, get_indexed_files

    has_session = bool(session_id and session_id in session_storage)
    registry = get_all_files()
    
    # If no specific files requested, audit all indexed files
    if not file_ids:
        file_ids = [fid for fid, fdata in registry.items() if fdata.get("indexed")]

    has_files = bool(file_ids)

    dataset_answer = None
    if has_session:
        try:
            dataset_answer = run_analyze_pipeline(
                session_id=session_id,
                user_query="Analyse l'ensemble du dataset pour un audit global.",
                mode=mode,
                model_name=model_name,
                metadata=metadata,
                sample_rows=sample_rows,
            )
        except Exception as e:
            dataset_answer = {"error": str(e)}

    document_intelligence = None
    file_sources = []
    if has_files:
        try:
            # We use RAG to extract high level insights from the selected files
            rag_res = execute_rag(
                query="Résume les points clés, les risques et les objectifs de ces documents pour un audit global.",
                model_name=model_name,
                file_ids=file_ids
            )
            document_intelligence = {
                "answer": rag_res.get("answer", ""),
                "limitations": rag_res.get("limitations", [])
            }
            file_sources = rag_res.get("sources", [])
        except Exception as e:
            document_intelligence = {"error": str(e)}

    # Run the Audit Agent
    audit_report = run_audit_agent(dataset_answer, document_intelligence, model_name=model_name)
    audit_report["sources"] = file_sources
    audit_report["performance"] = {"total_time_sec": round(time.time() - start, 2)}

    return audit_report
