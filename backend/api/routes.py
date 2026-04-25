from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List, Any
import pandas as pd
import io
import uuid
import json
from sqlalchemy.orm import Session

from backend.tools.smart_metadata import get_smart_metadata
from backend.core.config import settings
from backend.core.storage import session_storage
from backend.core.database import get_db
from backend.core.models import SessionModel, MessageModel

router = APIRouter()


# ──────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id: str
    user_query: str
    mode: str = "deep"  # "fast" or "deep"
    model_name: Optional[str] = None
    # Optionnel : le frontend peut fournir metadata/sample directement
    metadata: Optional[dict] = None
    sample_rows: Optional[List[Any]] = None


# ──────────────────────────────────────────────
# Endpoints existants (inchangés)
# ──────────────────────────────────────────────

@router.post("/upload")
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Format de fichier non supporté.")

    contents = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture : {str(e)}")

    df = df.dropna(how='all')
    metadata = get_smart_metadata(df)
    session_id = str(uuid.uuid4())

    new_session = SessionModel(
        id=session_id,
        filename=file.filename,
        metadata_json=metadata
    )
    db.add(new_session)
    db.commit()

    session_storage[session_id] = {
        "dataframe": df,
        "metadata": metadata,
        "filename": file.filename
    }

    from backend.core.indexer import index_session_data
    background_tasks.add_task(index_session_data, session_id, df, metadata)

    return {
        "message": f"Fichier {file.filename} analysé. RAG en cours.",
        "session_id": session_id,
        "metadata": metadata
    }


@router.post("/chat")
async def chat_with_agent(session_id: str, prompt: str, db: Session = Depends(get_db)):
    if session_id not in session_storage:
        session_db = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session_db:
            raise HTTPException(status_code=404, detail="Session introuvable.")
        else:
            raise HTTPException(status_code=400, detail="Dataset doit être ré-uploadé.")

    user_msg = MessageModel(session_id=session_id, role="user", content=prompt)
    db.add(user_msg)
    db.commit()

    from backend.orchestrator import run_orchestrator
    result = run_orchestrator(session_id, prompt, mode="chat")

    assistant_msg = MessageModel(
        session_id=session_id,
        role="assistant",
        content=result.get("answer", ""),
        plots=result.get("plots", [])
    )
    db.add(assistant_msg)
    db.commit()

    return result


@router.post("/profile")
async def get_data_profile(session_id: str):
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    from backend.orchestrator import run_orchestrator
    result = run_orchestrator(session_id, "Génère un profil complet et un audit de santé du dataset.", mode="profile")
    return result


# ──────────────────────────────────────────────
# Nouvel endpoint /analyze
# ──────────────────────────────────────────────

@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Pipeline multi-agents complet : Planner → Analyst → Reviewer → Reporter.

    Retourne un JSON structuré avec :
    - plan       : type de tâche + étapes
    - analysis   : insights, anomalies, corrélations
    - critic     : problèmes, limitations, confiance
    - report     : résumé + réponse finale
    - agent_trace: historique des agents
    """
    if request.session_id not in session_storage:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' introuvable. Uploadez d'abord un fichier."
        )

    from backend.orchestrator import run_analyze_pipeline

    result = run_analyze_pipeline(
        session_id=request.session_id,
        user_query=request.user_query,
        mode=request.mode,
        model_name=request.model_name,
        metadata=request.metadata,
        sample_rows=request.sample_rows
    )

    return result


# ──────────────────────────────────────────────
# Endpoints utilitaires
# ──────────────────────────────────────────────

@router.get("/sessions")
def get_all_sessions(db: Session = Depends(get_db)):
    return db.query(SessionModel).order_by(SessionModel.created_at.desc()).all()


@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    return (
        db.query(MessageModel)
        .filter(MessageModel.session_id == session_id)
        .order_by(MessageModel.timestamp.asc())
        .all()
    )


@router.get("/health")
def health_check():
    return {"status": "online", "active_sessions": len(session_storage)}


@router.get("/models")
def get_available_models():
    """Renvoie la liste des modèles disponibles configurés."""
    return {"models": settings.AVAILABLE_MODELS, "default": settings.DEFAULT_MODEL}


@router.get("/benchmark-models")
def run_model_benchmark():
    """Lance le benchmark sur tous les modèles."""
    from backend.model_benchmark import benchmark_model
    results = []
    for model in settings.AVAILABLE_MODELS:
        res = benchmark_model(model)
        results.append(res)
    return {"status": "success", "results": results}
