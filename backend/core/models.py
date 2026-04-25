from sqlalchemy import Column, String, DateTime, JSON, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import uuid

Base = declarative_base()

class SessionModel(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_json = Column(JSON) # Stockage des métadonnées du DataFrame
    
    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")

class MessageModel(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String) # user or assistant
    content = Column(Text)
    plots = Column(JSON) # JSON string array for plotly charts
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    session = relationship("SessionModel", back_populates="messages")
