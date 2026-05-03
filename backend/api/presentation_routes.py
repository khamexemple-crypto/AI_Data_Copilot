from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from backend.services.voice_presenter import generate_voice_presentation

router = APIRouter(prefix="/api/presentation", tags=["presentation"])


class VoicePresentationRequest(BaseModel):
    topic: str
    source_kind: str = "auto"
    session_id: Optional[str] = None
    file_ids: List[str] = []
    user_context: str = ""
    language: str = "fr-FR"
    tone: str = "professional"
    duration_minutes: int = 4
    model_name: Optional[str] = None


@router.post("/generate")
def create_voice_presentation(request: VoicePresentationRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="A topic is required.")
    try:
        return generate_voice_presentation(
            topic=request.topic.strip(),
            source_kind=request.source_kind,
            session_id=request.session_id,
            file_ids=request.file_ids,
            user_context=request.user_context,
            language=request.language,
            tone=request.tone,
            duration_minutes=max(1, min(request.duration_minutes, 12)),
            model_name=request.model_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
