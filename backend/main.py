from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.routes import router
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
