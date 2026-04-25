from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.core.models import Base
import os

DATABASE_URL = "sqlite:///./ai_copilot.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initialise la base de données (crée les tables si inexistantes).
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Dependency pour FastAPI permettant d'obtenir une session DB par requête.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
