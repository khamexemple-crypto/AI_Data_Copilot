from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from datetime import datetime
from backend.services.notebook_generator import generate_analysis_notebook

router = APIRouter(prefix="/api/notebook", tags=["Notebook"])


class NotebookRequest(BaseModel):
    session_id: str
    dataset_name: str = "Dataset"
    dataset_path: str = "data/dataset.csv"
    target_column: str = "target"
    dataset_summary: Optional[Dict] = {}
    data_quality: Optional[Dict] = {}
    insights: Optional[List[str]] = []
    visualizations: Optional[List[Dict]] = []
    ml_results: Optional[Dict] = {}
    xai_results: Optional[Dict] = {}
    recommendations: Optional[Any] = {}


@router.post("/generate")
async def create_notebook(req: NotebookRequest):
    """Generates a .ipynb notebook and returns the file path."""
    try:
        bundle = {
            "dataset_name": req.dataset_name,
            "dataset_path": req.dataset_path,
            "target_column": req.target_column,
            "dataset_summary": req.dataset_summary,
            "data_quality": req.data_quality,
            "insights": req.insights,
            "visualizations": req.visualizations,
            "ml_results": req.ml_results,
            "xai_results": req.xai_results,
            "recommendations": req.recommendations,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("data", f"notebook_{req.session_id}_{timestamp}.ipynb")

        path = generate_analysis_notebook(bundle, output_path)

        return {"status": "success", "notebook_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notebook generation failed: {str(e)}")


@router.post("/download")
async def download_notebook(req: NotebookRequest):
    """Generates a .ipynb and returns it as a downloadable file."""
    try:
        bundle = {
            "dataset_name": req.dataset_name,
            "dataset_path": req.dataset_path,
            "target_column": req.target_column,
            "dataset_summary": req.dataset_summary,
            "data_quality": req.data_quality,
            "insights": req.insights,
            "visualizations": req.visualizations,
            "ml_results": req.ml_results,
            "xai_results": req.xai_results,
            "recommendations": req.recommendations,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("data", f"notebook_{req.session_id}_{timestamp}.ipynb")

        path = generate_analysis_notebook(bundle, output_path)

        return FileResponse(
            path,
            media_type="application/x-ipynb+json",
            filename=os.path.basename(path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
