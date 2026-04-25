import hashlib
import json
from typing import Optional, Dict, Any

# In-memory dictionary to store cached responses
# Structure: { "hash_string": response_dict }
_analysis_cache: Dict[str, Dict[str, Any]] = {}

def generate_cache_key(user_query: str, mode: str, metadata: dict, sample_rows: list) -> str:
    """
    Génère une clé de cache stable et unique basée sur les paramètres de la requête.
    Utilise SHA-256 pour hasher une représentation JSON triée.
    """
    # Create a unified dictionary of all inputs to hash
    cache_payload = {
        "query": user_query.strip().lower(),
        "mode": mode,
        "metadata": metadata,
        "sample_rows": sample_rows
    }
    
    # Sort keys to ensure the same dict always produces the same string
    json_string = json.dumps(cache_payload, sort_keys=True)
    
    # Generate SHA-256 hash
    return hashlib.sha256(json_string.encode('utf-8')).hexdigest()

def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Récupère une réponse en cache si elle existe.
    Retourne None si non trouvée.
    """
    return _analysis_cache.get(cache_key)

def save_cached_response(cache_key: str, response: Dict[str, Any]) -> None:
    """
    Sauvegarde une réponse (si elle a été un succès) dans le cache.
    """
    if response.get("status") in ["success", "partial_success"]:
        # Deep copy to prevent accidental mutations
        _analysis_cache[cache_key] = json.loads(json.dumps(response))

def clear_cache() -> None:
    """Vide entièrement le cache en mémoire."""
    _analysis_cache.clear()
