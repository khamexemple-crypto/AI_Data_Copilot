from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from backend.services.analysis_memory import (
    save_analysis_session, 
    load_analysis_session, 
    list_analysis_sessions, 
    find_latest_session
)

router = APIRouter(prefix="/api/memory", tags=["Memory"])

class SessionPayload(BaseModel):
    session_id: str
    session_data: Dict[str, Any]

@router.post("/save")
async def save_session(payload: SessionPayload):
    try:
        save_analysis_session(payload.session_id, payload.session_data)
        return {"status": "success", "message": f"Session {payload.session_id} saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/load/{session_id}")
async def load_session(session_id: str):
    data = load_analysis_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return data

@router.get("/list")
async def list_sessions():
    """Returns metadata for all past sessions (to build a history sidebar)."""
    return list_analysis_sessions()

@router.get("/latest")
async def get_latest_session():
    data = find_latest_session()
    if not data:
        raise HTTPException(status_code=404, detail="No previous sessions found.")
    return data
