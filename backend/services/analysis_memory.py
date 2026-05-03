import sqlite3
import json
import os
from datetime import datetime

# Define path to the local SQLite DB.
# Assumes execution from the root workspace directory.
DB_PATH = os.path.join(os.getcwd(), "ai_copilot.db")

def init_db():
    """Initializes the SQLite database with the sessions table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            session_id TEXT PRIMARY KEY,
            dataset_name TEXT,
            analysis_summary TEXT,
            model_summary TEXT,
            full_state JSON,
            updated_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Ensure table exists when module loads
init_db()

def save_analysis_session(session_id: str, session_data: dict) -> None:
    """
    Saves or updates an analysis session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    dataset_name = session_data.get("dataset_name", "Unknown Dataset")
    analysis_summary = session_data.get("analysis_summary", "")
    model_summary = session_data.get("model_summary", "")
    
    full_state_json = json.dumps(session_data)
    updated_at = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO analysis_sessions (session_id, dataset_name, analysis_summary, model_summary, full_state, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            dataset_name=excluded.dataset_name,
            analysis_summary=excluded.analysis_summary,
            model_summary=excluded.model_summary,
            full_state=excluded.full_state,
            updated_at=excluded.updated_at
    ''', (session_id, dataset_name, analysis_summary, model_summary, full_state_json, updated_at))
    
    conn.commit()
    conn.close()

def load_analysis_session(session_id: str) -> dict:
    """
    Loads a specific analysis session by ID.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT full_state FROM analysis_sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return {}

def list_analysis_sessions() -> list:
    """
    Returns a list of all saved sessions with their metadata (excluding the heavy full_state JSON).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, dataset_name, analysis_summary, model_summary, updated_at FROM analysis_sessions ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": row[0],
            "dataset_name": row[1],
            "analysis_summary": row[2],
            "model_summary": row[3],
            "updated_at": row[4]
        }
        for row in rows
    ]

def find_latest_session() -> dict:
    """
    Returns the most recently updated session.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT full_state FROM analysis_sessions ORDER BY updated_at DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return {}
