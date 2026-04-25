import os

class Config:
    PROJECT_NAME: str = "AI Data Copilot API"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:lite")
    
    AVAILABLE_MODELS = [
        "deepseek-coder-v2:lite",
        "llama3.2:3b",
        "qwen2.5:3b",
        "phi3:mini",
        "mistral:7b"
    ]
    
    MODEL_REGISTRY = {
        "deepseek-coder-v2:lite": {"speed": "Medium", "quality": "High", "usage": "Code-heavy tasks, deep analysis"},
        "llama3.2:3b": {"speed": "Fast", "quality": "Medium", "usage": "General data analysis, fast summaries"},
        "qwen2.5:3b": {"speed": "Fast", "quality": "Medium", "usage": "General data analysis, fallback"},
        "phi3:mini": {"speed": "Very Fast", "quality": "Basic", "usage": "Fast mode, simple questions"},
        "mistral:7b": {"speed": "Medium", "quality": "High", "usage": "Balanced tasks, complex reasoning"}
    }
    
    @classmethod
    def get_recommended_model(cls, mode: str) -> str:
        if mode == "fast":
            return "llama3.2:3b" # Default fast
        return cls.DEFAULT_MODEL

settings = Config()
