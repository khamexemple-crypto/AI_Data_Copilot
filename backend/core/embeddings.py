from langchain_ollama import OllamaEmbeddings
from backend.core.config import settings

def get_embeddings():
    """
    Initialise le moteur d'embeddings utilisant Ollama.
    Modèle par défaut: mxbai-embed-large
    """
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=settings.OLLAMA_BASE_URL
    )
