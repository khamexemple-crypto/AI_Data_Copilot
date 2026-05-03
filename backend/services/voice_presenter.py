"""
Voice presenter agent.

Generates a spoken presentation script from dataset metadata, indexed documents,
or free-form context. Audio playback is handled by the frontend via browser
speech synthesis; this service owns the LLM script and persistence.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from backend.core.storage import session_storage
from backend.llm import call_llm, safe_json_parse
from backend.services.storage_hub import get_storage_hub, utc_now
from backend.storage import file_registry

logger = logging.getLogger(__name__)


VOICE_PRESENTER_SYSTEM = """You are the Voice Presenter Agent for AI Data Copilot.

Your job is to prepare a spoken presentation script that a browser TTS voice can read aloud.

Rules:
1. Use only the provided context. Do not invent facts.
2. Write in the requested language.
3. The tone must be clear, professional, and oral. Short sentences. Natural transitions.
4. Return ONLY valid JSON. No markdown. No extra text.
5. Every segment must be understandable when heard without seeing the screen.

Return this exact schema:
{
  "title": "Presentation title",
  "executive_summary": "2 sentence spoken summary.",
  "segments": [
    {
      "title": "Opening",
      "narration": "Text to speak aloud.",
      "duration_hint_sec": 30,
      "source_refs": ["source name"]
    }
  ],
  "closing": "Short spoken conclusion.",
  "limitations": ["Any missing context"]
}
"""


def _compact_json(data: Any, max_chars: int = 2500) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _collect_dataset_context(session_id: str | None) -> dict | None:
    if not session_id or session_id not in session_storage:
        return None
    stored = session_storage[session_id]
    metadata = stored.get("metadata", {})
    return {
        "session_id": session_id,
        "filename": stored.get("filename"),
        "shape": metadata.get("shape"),
        "columns": metadata.get("columns", [])[:40],
        "types": metadata.get("types", {}),
        "missing_values": metadata.get("missing_values", {}),
    }


def _collect_file_context(file_ids: list[str] | None, include_all: bool = False) -> list[dict]:
    registry = file_registry.get_all_files()
    if file_ids is None:
        selected_ids = [
            file_id for file_id, entry in registry.items()
            if include_all and entry.get("indexed")
        ]
    else:
        selected_ids = file_ids
    docs = []
    for file_id in selected_ids:
        entry = registry.get(file_id)
        if not entry:
            continue
        docs.append({
            "file_id": file_id,
            "filename": entry.get("filename"),
            "indexed": entry.get("indexed"),
            "summary": entry.get("summary"),
            "tags": entry.get("tags", []),
            "key_topics": entry.get("key_topics", []),
            "suggested_questions": entry.get("suggested_questions", []),
        })
    return docs


def build_voice_context(
    *,
    session_id: str | None = None,
    file_ids: list[str] | None = None,
    user_context: str | None = None,
    include_all_files: bool = False,
) -> dict:
    return {
        "dataset": _collect_dataset_context(session_id),
        "documents": _collect_file_context(file_ids, include_all=include_all_files),
        "user_context": user_context or "",
    }


def _allowed_sources(context: dict) -> list[str]:
    sources = []
    dataset = context.get("dataset")
    if dataset and dataset.get("filename"):
        sources.append(dataset["filename"])
    for doc in context.get("documents", []):
        if doc.get("filename"):
            sources.append(doc["filename"])
        if doc.get("file_id"):
            sources.append(doc["file_id"])
    if context.get("user_context"):
        sources.append("user_context")
    return sources


def _uses_allowed_sources(parsed: dict, allowed_sources: list[str]) -> bool:
    if not allowed_sources:
        return True
    allowed = set(allowed_sources)
    refs = []
    for segment in parsed.get("segments", []):
        if isinstance(segment, dict):
            refs.extend(str(ref) for ref in segment.get("source_refs", []))
    return bool(refs) and all(ref in allowed for ref in refs)


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "dans", "pour", "avec",
        "les", "des", "une", "un", "est", "sont", "nous", "vous", "cela",
        "cette", "ces", "sur", "par", "pas", "plus", "aux", "qui", "que",
        "bonjour", "merci", "aujourd", "hui",
    }
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]{4,}", (text or "").lower())
    return {token for token in tokens if token not in stopwords}


def _custom_context_is_grounded(parsed: dict, context: dict) -> bool:
    if context.get("documents") or context.get("dataset") or not context.get("user_context"):
        return True

    context_tokens = _meaningful_tokens(context["user_context"])
    if len(context_tokens) < 4:
        return True

    for segment in parsed.get("segments", []):
        if not isinstance(segment, dict):
            continue
        narration_tokens = _meaningful_tokens(segment.get("narration", ""))
        if len(narration_tokens) < 5:
            continue
        overlap = len(narration_tokens & context_tokens) / max(len(narration_tokens), 1)
        new_tokens = narration_tokens - context_tokens
        if overlap < 0.35 and len(new_tokens) > 8:
            return False
    return True


def _normalise_segments(raw_segments: Any, title: str) -> list[dict]:
    if not isinstance(raw_segments, list):
        raw_segments = []

    segments = []
    for index, segment in enumerate(raw_segments[:8], start=1):
        if not isinstance(segment, dict):
            continue
        narration = str(segment.get("narration", "")).strip()
        if not narration:
            continue
        segments.append({
            "title": str(segment.get("title") or f"Partie {index}").strip(),
            "narration": narration,
            "duration_hint_sec": int(segment.get("duration_hint_sec") or 45),
            "source_refs": [
                str(source) for source in segment.get("source_refs", [])
                if str(source).strip()
            ],
        })

    if segments:
        return segments

    return [{
        "title": title,
        "narration": "Je n'ai pas assez de contexte fiable pour construire une presentation detaillee.",
        "duration_hint_sec": 20,
        "source_refs": [],
    }]


def _fallback_presentation(topic: str, context: dict, language: str) -> dict:
    docs = context.get("documents", [])
    dataset = context.get("dataset")
    segments = [{
        "title": "Introduction",
        "narration": f"Bonjour. Je vais presenter les informations disponibles sur: {topic}.",
        "duration_hint_sec": 20,
        "source_refs": [],
    }]

    if dataset:
        segments.append({
            "title": "Dataset",
            "narration": (
                f"Le dataset {dataset.get('filename')} contient {dataset.get('shape')} "
                f"et les colonnes principales sont: {', '.join(dataset.get('columns', [])[:10])}."
            ),
            "duration_hint_sec": 35,
            "source_refs": [dataset.get("filename") or "dataset"],
        })

    for doc in docs[:4]:
        summary = doc.get("summary") or "resume non disponible"
        segments.append({
            "title": doc.get("filename") or "Document",
            "narration": f"Pour le document {doc.get('filename')}, voici le point essentiel: {summary}",
            "duration_hint_sec": 40,
            "source_refs": [doc.get("filename") or doc.get("file_id")],
        })

    if context.get("user_context"):
        segments.append({
            "title": "Contexte demande",
            "narration": context["user_context"][:700],
            "duration_hint_sec": 35,
            "source_refs": ["user_context"],
        })

    segments.append({
        "title": "Conclusion",
        "narration": "En conclusion, cette presentation reprend uniquement les informations disponibles dans l'application.",
        "duration_hint_sec": 20,
        "source_refs": [],
    })

    return {
        "title": topic or "Presentation orale",
        "executive_summary": "Presentation generee en mode degrade avec le contexte disponible.",
        "segments": segments,
        "closing": "Merci pour votre ecoute.",
        "limitations": ["LLM unavailable or returned invalid JSON."],
        "language": language,
        "fallback_used": True,
    }


def _build_prompt(topic: str, context: dict, language: str, tone: str, duration_minutes: int) -> str:
    allowed_sources = _allowed_sources(context)
    return (
        f"Topic: {topic}\n"
        f"Language: {language}\n"
        f"Tone: {tone}\n"
        f"Target duration: {duration_minutes} minutes\n\n"
        f"Allowed source_refs: {json.dumps(allowed_sources, ensure_ascii=False)}\n"
        "Every segment source_refs must use ONLY values from Allowed source_refs.\n"
        "If the context is too small, keep the presentation short instead of adding outside knowledge.\n\n"
        "Context:\n"
        f"{_compact_json(context, max_chars=6500)}\n\n"
        "Create the spoken presentation now. Return only JSON."
    )


def generate_voice_presentation(
    *,
    topic: str,
    source_kind: str = "auto",
    session_id: str | None = None,
    file_ids: list[str] | None = None,
    user_context: str | None = None,
    language: str = "fr-FR",
    tone: str = "professional",
    duration_minutes: int = 4,
    model_name: str | None = None,
) -> dict:
    include_dataset = source_kind in {"auto", "dataset"} and bool(session_id)
    include_all_files = source_kind in {"auto", "files"} and not file_ids
    selected_file_ids = file_ids if source_kind in {"auto", "files"} or file_ids else []

    context = build_voice_context(
        session_id=session_id if include_dataset else None,
        file_ids=selected_file_ids,
        user_context=user_context,
        include_all_files=include_all_files,
    )
    parsed = None
    fallback_used = False

    try:
        raw = call_llm(
            prompt=_build_prompt(topic, context, language, tone, duration_minutes),
            system=VOICE_PRESENTER_SYSTEM,
            timeout=120,
            model_name=model_name,
        )
        parsed = safe_json_parse(raw)
    except Exception as exc:
        logger.warning("Voice presenter LLM failed: %s", exc)

    if not isinstance(parsed, dict) or not parsed.get("segments"):
        parsed = _fallback_presentation(topic, context, language)
        fallback_used = True
    elif not _uses_allowed_sources(parsed, _allowed_sources(context)):
        logger.warning("Voice presenter rejected LLM output with invalid source_refs.")
        parsed = _fallback_presentation(topic, context, language)
        parsed.setdefault("limitations", []).append("LLM output rejected because source_refs were not grounded.")
        fallback_used = True
    elif not _custom_context_is_grounded(parsed, context):
        logger.warning("Voice presenter rejected LLM output with weak custom-context grounding.")
        parsed = _fallback_presentation(topic, context, language)
        parsed.setdefault("limitations", []).append("LLM output rejected because custom-context grounding was too weak.")
        fallback_used = True

    title = str(parsed.get("title") or topic or "Presentation orale").strip()
    segments = _normalise_segments(parsed.get("segments"), title)
    closing = str(parsed.get("closing") or "Merci pour votre ecoute.").strip()
    speech_text = "\n\n".join(
        [str(parsed.get("executive_summary") or title)]
        + [segment["narration"] for segment in segments]
        + [closing]
    )

    presentation_id = str(uuid.uuid4())
    storage_external_id = f"presentation-{presentation_id}"
    presentation = {
        "presentation_id": presentation_id,
        "storage_external_id": storage_external_id,
        "topic": topic,
        "title": title,
        "source_kind": source_kind,
        "source_ref": session_id or ",".join(file_ids or []),
        "executive_summary": str(parsed.get("executive_summary") or ""),
        "segments": segments,
        "closing": closing,
        "speech_text": speech_text,
        "voice_profile": {
            "language": language,
            "tone": tone,
            "rate": 1.0,
            "pitch": 1.0,
        },
        "metadata": {
            "duration_minutes": duration_minutes,
            "fallback_used": fallback_used or bool(parsed.get("fallback_used")),
            "context_summary": {
                "has_dataset": bool(context.get("dataset")),
                "document_count": len(context.get("documents", [])),
                "has_user_context": bool(context.get("user_context")),
            },
        },
        "limitations": parsed.get("limitations", []),
        "created_at": utc_now(),
    }

    hub = get_storage_hub()
    hub.store_artifact(
        object_type="voice_presentation",
        title=title,
        content=json.dumps(presentation, indent=2, ensure_ascii=False),
        extension="json",
        external_id=storage_external_id,
        tags=["voice", "presentation", language],
        metadata=presentation["metadata"],
        created_by_agent="voice_presenter",
    )
    hub.record_presentation(presentation)
    return presentation
