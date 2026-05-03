from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.routes import router
from backend.api.database_routes import router as database_router
from backend.api.memory_routes import router as memory_router
from backend.api.notebook_routes import router as notebook_router
from backend.api.report_routes import router as report_router
from backend.api.storage_routes import router as storage_router
from backend.api.presentation_routes import router as presentation_router
from backend.core.config import settings
from backend.core.database import init_db
from backend.llm import warm_up_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialisation de la BDD
    init_db()
    # Warm-up du LLM (non bloquant si erreur)
    warm_up_llm()
    yield

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Autoriser le CORS pour que Streamlit puisse communiquer avec FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(router, prefix="/api")
app.include_router(database_router)   # prefix="/api/database" defined internally
app.include_router(memory_router)     # prefix="/api/memory" defined internally
app.include_router(notebook_router)   # prefix="/api/notebook" defined internally
app.include_router(report_router)     # prefix="/api/report" defined internally
app.include_router(storage_router)
app.include_router(presentation_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
