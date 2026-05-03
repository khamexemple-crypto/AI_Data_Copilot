from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.services.storage_hub import get_storage_hub

router = APIRouter(prefix="/api/storage", tags=["storage"])


class StatusUpdateRequest(BaseModel):
    status: str
    metadata: dict = {}
    actor: str = "api"


@router.get("/metrics")
def storage_metrics():
    try:
        return get_storage_hub().metrics()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/objects")
def list_storage_objects(object_type: Optional[str] = None, status: Optional[str] = None):
    try:
        return {"objects": get_storage_hub().list_objects(object_type=object_type, status=status)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/objects/{external_id}")
def get_storage_object(external_id: str):
    try:
        item = get_storage_hub().get_object(external_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not item:
        raise HTTPException(status_code=404, detail="Storage object not found")
    return item


@router.post("/objects/{external_id}/status")
def update_storage_status(external_id: str, request: StatusUpdateRequest):
    try:
        get_storage_hub().set_status(
            external_id,
            request.status,
            metadata=request.metadata,
            actor=request.actor,
        )
        return {"status": "success", "external_id": external_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Storage object not found")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/events")
def list_storage_events(external_id: Optional[str] = None, limit: int = 100):
    try:
        return {"events": get_storage_hub().list_events(external_id=external_id, limit=min(limit, 500))}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
