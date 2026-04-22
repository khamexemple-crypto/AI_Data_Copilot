import os

class Config:
    PROJECT_NAME: str = "AI Data Copilot API"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:lite")

settings = Config()
