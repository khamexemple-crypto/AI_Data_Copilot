from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from backend.services.report_generator import generate_report

router = APIRouter(prefix="/api/report", tags=["Reporting"])

class ReportRequest(BaseModel):
    session_id: str
    output_format: str = "markdown"
    dataset_summary: Optional[Dict] = {}
    analysis_results: Optional[Dict] = {}
    visualization_summaries: Optional[Dict] = {}
    ml_results: Optional[Dict] = {}
    xai_results: Optional[Dict] = {}
    recommendations: Optional[Dict] = {}

@router.post("/generate")
async def create_report(req: ReportRequest):
    try:
        # In production, replace `req.dataset_summary` etc. with data retrieved from 
        # your database or LangGraph state using the `req.session_id`.
        
        result = generate_report(
            dataset_summary=req.dataset_summary,
            analysis_results=req.analysis_results,
            visualization_summaries=req.visualization_summaries,
            ml_results=req.ml_results,
            xai_results=req.xai_results,
            recommendations=req.recommendations,
            output_format=req.output_format
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
