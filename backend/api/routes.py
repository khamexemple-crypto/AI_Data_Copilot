from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List, Any
import pandas as pd
import io
import uuid
import json
import logging
from sqlalchemy.orm import Session

from backend.tools.smart_metadata import get_smart_metadata
from backend.core.config import settings
from backend.core.storage import session_storage
from backend.core.database import get_db
from backend.core.models import SessionModel, MessageModel
from backend.storage import file_manager, file_registry
from backend.rag import document_loader, chunker, vector_store, keyword_index
from backend.rag.file_summarizer import generate_file_intelligence
from backend.rag.rag_pipeline import execute_rag
from backend.services.storage_hub import get_storage_hub, sync_uploaded_file

router = APIRouter()
logger = logging.getLogger(__name__)


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


# ──────────────────────────────────────────────
# FILE STORAGE & RAG ENDPOINTS
# ──────────────────────────────────────────────

@router.post("/files/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and store a file, registering it in the metadata registry."""
    file_id, file_path = file_manager.save_uploaded_file(file)
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'txt'
    file_registry.register_file(file_id, file.filename, ext)
    sync_uploaded_file(
        file_id=file_id,
        filename=file.filename,
        file_type=ext,
        file_path=file_path,
        status="uploaded",
        metadata={"source": "files/upload"},
    )
    return {"status": "success", "file_id": file_id, "filename": file.filename, "file_type": ext}


@router.get("/files")
def list_files():
    """List all stored files and their metadata."""
    return file_registry.get_all_files()


@router.delete("/files/{file_id}")
def delete_stored_file(file_id: str):
    """Delete a stored file from disk and registry."""
    registry = file_registry.get_all_files()
    if file_id not in registry:
        raise HTTPException(status_code=404, detail="File not found")

    index_cleanup = {"vector": True, "keyword": True}
    try:
        vector_store.delete_file_chunks(file_id)
    except Exception as e:
        index_cleanup["vector"] = False
        logger.warning("delete_stored_file: vector cleanup failed for %s: %s", file_id, e)

    try:
        keyword_index.keyword_store.remove_file(file_id)
    except Exception as e:
        index_cleanup["keyword"] = False
        logger.warning("delete_stored_file: keyword cleanup failed for %s: %s", file_id, e)

    try:
        get_storage_hub().delete_object(file_id)
    except Exception as e:
        logger.warning("delete_stored_file: storage hub cleanup failed for %s: %s", file_id, e)

    file_manager.delete_file(file_id, registry[file_id]["filename"])
    file_registry.delete_file_from_registry(file_id)
    return {"status": "success", "deleted": file_id, "index_cleanup": index_cleanup}


@router.get("/files/{file_id}/intelligence")
def get_file_intelligence(file_id: str):
    """
    Return the auto-generated intelligence for a single indexed file:
    summary, tags, key_topics, suggested_questions.
    """
    intel = file_registry.get_file_intelligence(file_id)
    if not intel:
        raise HTTPException(status_code=404, detail="File not found")
    if not intel.get("indexed"):
        raise HTTPException(status_code=400, detail="File not yet indexed. Call /files/index first.")
    return intel


@router.post("/files/index")
async def index_file(file_id: str, model_name: Optional[str] = None):
    """
    Extract text, chunk it, index into Vector DB and BM25, 
    and generate automatic file intelligence (summary, tags, questions).
    """
    registry = file_registry.get_all_files()
    if file_id not in registry: 
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = registry[file_id]["filename"]
    path = file_manager.get_file_path(file_id, filename)
    
    # 1. Extraction & Indexing
    try:
        text = document_loader.extract_text(path, filename)
    except Exception as e:
        logger.exception("index_file: extraction crashed for %s (%s)", filename, file_id)
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed for {filename}: {e}",
        )

    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text found. If this is a scanned PDF, verify that "
                "Tesseract OCR and the requested OCR languages are installed."
            ),
        )

    chunks = chunker.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks generated from extracted text.")

    try:
        vector_store.add_chunks_to_store(file_id, filename, chunks)
    except Exception as e:
        logger.exception("index_file: vector indexing failed for %s (%s)", filename, file_id)
        raise HTTPException(
            status_code=503,
            detail=f"Vector indexing failed for {filename}: {e}",
        )
    
    # Simple keyword indexing (passing the same chunks)
    metas = [{"file_id": file_id, "filename": filename, "chunk_id": str(i)} for i in range(len(chunks))]
    keyword_indexed = True
    try:
        keyword_index.keyword_store.add_chunks(chunks, metas)
    except Exception as e:
        keyword_indexed = False
        logger.warning("index_file: keyword indexing failed for %s (%s): %s", filename, file_id, e)
    
    # 2. Automatic File Intelligence
    intelligence = generate_file_intelligence(text, model_name)

    # 3. Update registry with intelligence + indexing status
    file_registry.update_file_metadata(file_id, {
        **intelligence,
        "indexed"       : True,
        "indexed_chunks": len(chunks),
    })
    try:
        get_storage_hub().set_status(
            file_id,
            "indexed",
            metadata={
                "indexed_chunks": len(chunks),
                "keyword_indexed": keyword_indexed,
                "intelligence": intelligence,
            },
            actor="files/index",
        )
        get_storage_hub().add_version(
            external_id=file_id,
            storage_uri=path,
            content_hash=None,
            size_bytes=None,
            metadata={"indexed_chunks": len(chunks), "stage": "indexed"},
            created_by_agent="rag_indexer",
        )
    except Exception as e:
        logger.warning("index_file: storage hub status sync failed for %s: %s", file_id, e)

    return {
        "status"         : "success",
        "file_id"        : file_id,
        "filename"       : filename,
        "indexed_chunks" : len(chunks),
        "keyword_indexed": keyword_indexed,
        "intelligence"   : intelligence,
    }


@router.post("/ask-files")
def ask_files(query: str, model_name: Optional[str] = None):
    """
    Query the intelligent file knowledge base using RAG.
    Includes Hybrid Search, Reranking, and Grounded Generation.
    """
    result = execute_rag(query, model_name, file_ids=None)
    return result


class CompareRequest(BaseModel):
    question: str
    file_ids: List[str]
    model_name: Optional[str] = None

@router.post("/compare-files")
def compare_documents(request: CompareRequest):
    """
    Compare multiple indexed documents to find similarities, differences, and contradictions.
    """
    from backend.rag.document_comparator import compare_files
    result = compare_files(request.question, request.file_ids, request.model_name)
    return result


# ──────────────────────────────────────────────
# UNIFIED /ask ENDPOINT
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    question    : str
    session_id  : Optional[str]       = None
    file_ids    : Optional[List[str]] = None
    mode        : str                 = "deep"
    model_name  : Optional[str]       = None
    metadata    : Optional[dict]      = None
    sample_rows : Optional[List[Any]] = None


@router.post("/ask")
async def unified_ask(request: AskRequest):
    """
    Unified intelligent endpoint.

    The Router Agent classifies the question and dispatches to:
      - dataset  → Planner → Analyst → Reviewer → Reporter
      - files    → Hybrid Search → Reranker → RAG Agent
      - mixed    → dataset pipeline + RAG pipeline → Mixed Reporter
      - general  → graceful message asking for data/documents

    Response always includes:
      status, route, router, final_answer, sources, limitations, agent_trace
    """
    from backend.orchestrator import run_unified_ask

    result = run_unified_ask(
        question    = request.question,
        session_id  = request.session_id,
        file_ids    = request.file_ids or [],
        mode        = request.mode,
        model_name  = request.model_name,
        metadata    = request.metadata,
        sample_rows = request.sample_rows,
    )
    return result


# ──────────────────────────────────────────────
# AUTO-AUDIT ENDPOINT
# ──────────────────────────────────────────────

class AuditRequest(BaseModel):
    session_id  : Optional[str]       = None
    file_ids    : Optional[List[str]] = None
    mode        : str                 = "deep"
    model_name  : Optional[str]       = None
    metadata    : Optional[dict]      = None
    sample_rows : Optional[List[Any]] = None

@router.post("/auto-audit")
async def auto_audit(request: AuditRequest):
    """
    Run the Auto-Audit agent on datasets and/or files.
    """
    from backend.orchestrator import run_auto_audit

    result = run_auto_audit(
        session_id  = request.session_id,
        file_ids    = request.file_ids or [],
        mode        = request.mode,
        model_name  = request.model_name,
        metadata    = request.metadata,
        sample_rows = request.sample_rows,
    )
    return result
