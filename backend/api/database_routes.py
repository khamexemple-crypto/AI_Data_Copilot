from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from backend.services.database_connector import DatabaseConnector
from backend.services.nl_to_sql import nl_to_sql

router = APIRouter(prefix="/api/database", tags=["Database"])

# ── Request / Response models ────────────────

class ConnectRequest(BaseModel):
    db_type: str = "sqlite"          # "sqlite" or "postgresql"
    db_path: str = ""                # SQLite file path
    connection_string: str = ""      # PostgreSQL DSN

class QueryRequest(BaseModel):
    db_type: str = "sqlite"
    db_path: str = ""
    connection_string: str = ""
    sql: str

class NLQueryRequest(BaseModel):
    db_type: str = "sqlite"
    db_path: str = ""
    connection_string: str = ""
    question: str


def _get_connector(req) -> DatabaseConnector:
    return DatabaseConnector(
        db_type=req.db_type,
        db_path=req.db_path,
        connection_string=req.connection_string,
    )


# ── Endpoints ────────────────────────────────

@router.post("/schema")
async def inspect_schema(req: ConnectRequest):
    """Returns all tables and their column schemas."""
    try:
        db = _get_connector(req)
        db.connect()
        schema = db.get_full_schema()
        db.close()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def run_query(req: QueryRequest):
    """Executes a raw (but safety-checked) SELECT query."""
    try:
        db = _get_connector(req)
        db.connect()
        result = db.execute_safe(req.sql)
        db.close()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_database(req: NLQueryRequest):
    """Translates a natural-language question into SQL, executes it, and returns the result."""
    try:
        db = _get_connector(req)
        db.connect()
        schema = db.get_full_schema()

        translation = nl_to_sql(req.question, schema)
        if "error" in translation:
            db.close()
            raise HTTPException(status_code=400, detail=translation["error"])

        sql = translation["sql"]
        result = db.execute_safe(sql)
        db.close()

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "question": req.question,
            "generated_sql": sql,
            "matched_pattern": translation.get("matched_pattern"),
            "table": translation.get("table"),
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
