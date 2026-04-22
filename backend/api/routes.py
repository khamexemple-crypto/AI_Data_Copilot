from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import uuid

from backend.tools.smart_metadata import get_smart_metadata
from backend.core.config import settings
from backend.core.storage import session_storage

router = APIRouter()

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Format de fichier non supporté.")

    # Lecture du fichier
    contents = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture : {str(e)}")

    # Nettoyage de base automatique
    df = df.dropna(how='all')
    
    # Extraction des métadonnées
    metadata = get_smart_metadata(df)
    
    # Création d'une session unique pour ce dataset
    session_id = str(uuid.uuid4())
    session_storage[session_id] = {
        "dataframe": df,
        "metadata": metadata,
        "filename": file.filename
    }
    
    return {
        "message": f"Fichier {file.filename} analysé avec succès",
        "session_id": session_id,
        "metadata": metadata
    }

@router.post("/chat")
async def chat_with_agent(session_id: str, prompt: str):
    """
    Endpoint pour interagir avec le système Multi-Agents.
    """
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session expirée ou introuvable. Veuillez re-uploader vos données.")
    
    # On délègue l'exécution à l'orchestrateur (sera implémenté)
    from backend.agents.orchestrator import run_orchestrator
    result = run_orchestrator(session_id, prompt)
    
    return result

@router.get("/health")
def health_check():
    return {
        "status": "online", 
        "model_configured": settings.OLLAMA_MODEL,
        "active_sessions": len(session_storage)
    }
