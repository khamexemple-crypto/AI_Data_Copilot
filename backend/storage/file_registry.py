"""
backend/storage/file_registry.py
----------------------------------
Persistent JSON file registry for uploaded documents.

Schema (per file_id)
────────────────────
{
    "filename"           : str,
    "type"               : str,
    "indexed"            : bool,
    "summary"            : str,       # set after indexing
    "tags"               : [str],     # set after indexing
    "key_topics"         : [str],     # set after indexing
    "suggested_questions": [str],     # set after indexing
    "indexed_chunks"     : int,       # set after indexing
    "indexed_at"         : str        # ISO timestamp
}
"""

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

REGISTRY_PATH = "data/file_registry.json"


# ── Low-level I/O ─────────────────────────────────────────────────────────────

def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("load_registry: failed to read registry — %s", e)
        return {}


def save_registry(data: dict) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logger.error("save_registry: failed to write registry — %s", e)


# ── Public helpers ────────────────────────────────────────────────────────────

def register_file(file_id: str, filename: str, file_type: str) -> None:
    """Register a newly uploaded file with default metadata."""
    registry = load_registry()
    registry[file_id] = {
        "filename"           : filename,
        "type"               : file_type,
        "indexed"            : False,
        "summary"            : None,
        "tags"               : [],
        "key_topics"         : [],
        "suggested_questions": [],
        "indexed_chunks"     : 0,
        "indexed_at"         : None,
    }
    save_registry(registry)
    logger.info("register_file: registered %s (%s)", filename, file_id)


def update_file_metadata(file_id: str, metadata: dict) -> bool:
    """
    Merge *metadata* into the registry entry for *file_id*.
    If file_id does not exist, logs a warning and returns False.
    Automatically sets indexed_at timestamp when indexed=True.
    """
    registry = load_registry()
    if file_id not in registry:
        logger.warning("update_file_metadata: file_id %s not in registry", file_id)
        return False

    registry[file_id].update(metadata)

    # Auto-stamp when we mark indexed
    if metadata.get("indexed") and not registry[file_id].get("indexed_at"):
        registry[file_id]["indexed_at"] = datetime.now(timezone.utc).isoformat()

    save_registry(registry)
    logger.info("update_file_metadata: updated %s", file_id)
    return True


def mark_indexed(file_id: str, chunk_count: int = 0) -> None:
    """Convenience: mark a file as indexed with optional chunk count."""
    update_file_metadata(file_id, {
        "indexed"       : True,
        "indexed_chunks": chunk_count,
    })


def get_file_intelligence(file_id: str) -> dict | None:
    """
    Return the intelligence metadata for a single file, or None if not found.

    Returns
    ───────
    dict:
        file_id, filename, summary, tags, key_topics, suggested_questions, indexed_at
    """
    registry = load_registry()
    entry = registry.get(file_id)
    if not entry:
        return None
    return {
        "file_id"            : file_id,
        "filename"           : entry.get("filename", ""),
        "summary"            : entry.get("summary") or "No summary available.",
        "tags"               : entry.get("tags", []),
        "key_topics"         : entry.get("key_topics", []),
        "suggested_questions": entry.get("suggested_questions", []),
        "indexed"            : entry.get("indexed", False),
        "indexed_chunks"     : entry.get("indexed_chunks", 0),
        "indexed_at"         : entry.get("indexed_at"),
    }


def delete_file_from_registry(file_id: str) -> bool:
    """Remove a file entry from the registry. Returns True on success."""
    registry = load_registry()
    if file_id not in registry:
        logger.warning("delete_file_from_registry: %s not found", file_id)
        return False
    del registry[file_id]
    save_registry(registry)
    logger.info("delete_file_from_registry: removed %s", file_id)
    return True


def get_all_files() -> dict:
    """Return the full registry dict (keyed by file_id)."""
    return load_registry()


def get_indexed_files() -> list[dict]:
    """Return a list of file intelligence dicts for all indexed files only."""
    registry = load_registry()
    return [
        {**{"file_id": fid}, **entry}
        for fid, entry in registry.items()
        if entry.get("indexed")
    ]
