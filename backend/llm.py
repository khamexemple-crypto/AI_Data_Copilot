"""
LLM wrapper — appels directs à Ollama via requests.
Pas de dépendance LangChain ici pour maximiser la stabilité.
"""

import json
import re
import requests
from backend.core.config import settings

class LLMTimeoutError(Exception):
    pass

class LLMConnectionError(Exception):
    pass

class LLMParsingError(Exception):
    pass

# ──────────────────────────────────────────────
# Appel LLM principal
# ──────────────────────────────────────────────

def call_llm(prompt: str, system: str = "", timeout: int = 120, model_name: str = None) -> str:
    """
    Envoie une requête à Ollama et retourne la réponse texte brute.
    En cas d'erreur de modèle introuvable, tente un fallback sur le modèle par défaut.
    """
    target_model = model_name if model_name else settings.DEFAULT_MODEL
    
    def _do_call(model: str):
        payload = {
            "model": model,
            "prompt": prompt if not system else f"{system}\n\n{prompt}",
            "stream": False,
        }
        if system:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        else:
            url = f"{settings.OLLAMA_BASE_URL}/api/generate"

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "message" in data:
            return data["message"]["content"].strip()
        return data.get("response", "").strip()

    try:
        return _do_call(target_model)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404 and target_model != settings.DEFAULT_MODEL:
            print(f"⚠️ [LLM] Modèle '{target_model}' non trouvé. Fallback sur '{settings.DEFAULT_MODEL}'.")
            return _do_call(settings.DEFAULT_MODEL)
        raise e
    except requests.exceptions.Timeout:
        print(f"⏱️ [LLM] Timeout après {timeout}s sur le modèle {target_model}")
        raise LLMTimeoutError(f"Le LLM a mis trop de temps à répondre (> {timeout}s).")
    except requests.exceptions.ConnectionError:
        print("❌ [LLM] Ollama inaccessible — vérifie que le serveur est lancé.")
        raise LLMConnectionError("Le serveur Ollama est inaccessible.")
    except Exception as e:
        print(f"❌ [LLM] Erreur inattendue avec {target_model} : {e}")
        raise e


# ──────────────────────────────────────────────
# Parser JSON robuste
# ──────────────────────────────────────────────

def safe_json_parse(text: str) -> dict | None:
    """
    Essaie de parser du JSON depuis une réponse LLM potentiellement bruitée.
    Stratégies :
      1. Parse direct
      2. Extraction du premier bloc ```json … ```
      3. Extraction du premier objet JSON { … }
    Retourne None si toutes les tentatives échouent.
    """
    if not text:
        return None

    # 1. Parse direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Bloc markdown ```json … ```
    md_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Premier { … } complet dans la chaîne
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None

# ──────────────────────────────────────────────
# Warm-Up
# ──────────────────────────────────────────────

def warm_up_llm(model_name: str = None):
    """
    Tente de réveiller le modèle localement (cold start).
    """
    target = model_name if model_name else settings.DEFAULT_MODEL
    print(f"🔥 [LLM] Tentative de warm-up du modèle Ollama ({target})...")
    try:
        res = call_llm(prompt="Reply with OK only.", timeout=15, model_name=target)
        print(f"✅ [LLM] Warm-up réussi pour {target}. Réponse: {res[:20]}")
    except LLMTimeoutError:
        print(f"⚠️ [LLM] Warm-up a provoqué un timeout pour {target}.")
    except LLMConnectionError:
        print("⚠️ [LLM] Ollama inaccessible pendant le warm-up.")
    except Exception as e:
        print(f"⚠️ [LLM] Erreur pendant le warm-up: {e}")
