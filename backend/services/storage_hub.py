"""
Storage Hub for AI Data Copilot.

PostgreSQL is the production target. Local JSON storage is kept as a
development/test fallback so the app still runs without a database server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = Path(os.getenv("STORAGE_LOCAL_DIR", "data/storage_hub"))
POSTGRES_DSN = os.getenv("POSTGRES_STORAGE_DSN") or os.getenv("STORAGE_DATABASE_URL")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("StorageHub: failed to load %s: %s", path, exc)
        return default


def atomic_json_save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def infer_mime_type(filename: str | None) -> str | None:
    if not filename:
        return None
    return mimetypes.guess_type(filename)[0]


class LocalStorageHub:
    def __init__(self, base_dir: str | Path = DEFAULT_STORAGE_DIR):
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "manifest.json"
        self.artifact_dir = self.base_dir / "artifacts"

    def _load(self) -> dict:
        return safe_json_load(self.manifest_path, {"objects": {}, "events": [], "presentations": {}})

    def _save(self, data: dict) -> None:
        atomic_json_save(self.manifest_path, data)

    def register_object(
        self,
        *,
        external_id: str,
        object_type: str,
        title: str,
        source_filename: str | None = None,
        mime_type: str | None = None,
        owner_session_id: str | None = None,
        status: str = "stored",
        storage_uri: str | None = None,
        content_hash: str | None = None,
        size_bytes: int | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        data = self._load()
        objects = data.setdefault("objects", {})
        existing = objects.get(external_id, {})
        now = utc_now()

        item = {
            **existing,
            "external_id": external_id,
            "object_type": object_type,
            "title": title,
            "source_filename": source_filename,
            "mime_type": mime_type,
            "owner_session_id": owner_session_id,
            "status": status,
            "storage_uri": storage_uri,
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "tags": tags or [],
            "metadata": {**existing.get("metadata", {}), **(metadata or {})},
            "versions": existing.get("versions", []),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        objects[external_id] = item
        data.setdefault("events", []).append({
            "event_id": str(uuid.uuid4()),
            "external_id": external_id,
            "event_type": "object_registered",
            "actor": "storage",
            "message": status,
            "payload": metadata or {},
            "created_at": now,
        })
        self._save(data)
        return item

    def add_version(
        self,
        *,
        external_id: str,
        storage_uri: str | None = None,
        content_hash: str | None = None,
        size_bytes: int | None = None,
        metadata: dict | None = None,
        created_by_agent: str = "system",
    ) -> dict:
        data = self._load()
        item = data.setdefault("objects", {}).get(external_id)
        if not item:
            raise KeyError(f"Unknown storage object: {external_id}")

        version = {
            "version_id": str(uuid.uuid4()),
            "version_number": len(item.get("versions", [])) + 1,
            "storage_uri": storage_uri,
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "metadata": metadata or {},
            "created_by_agent": created_by_agent,
            "created_at": utc_now(),
        }
        item.setdefault("versions", []).append(version)
        item["current_version_id"] = version["version_id"]
        item["storage_uri"] = storage_uri or item.get("storage_uri")
        item["content_hash"] = content_hash or item.get("content_hash")
        item["size_bytes"] = size_bytes if size_bytes is not None else item.get("size_bytes")
        item["metadata"] = {**item.get("metadata", {}), **(metadata or {})}
        item["updated_at"] = utc_now()
        data.setdefault("events", []).append({
            "event_id": str(uuid.uuid4()),
            "external_id": external_id,
            "event_type": "version_added",
            "actor": created_by_agent,
            "message": version["version_id"],
            "payload": metadata or {},
            "created_at": utc_now(),
        })
        self._save(data)
        return version

    def set_status(
        self,
        external_id: str,
        status: str,
        metadata: dict | None = None,
        actor: str = "system",
    ) -> None:
        data = self._load()
        item = data.setdefault("objects", {}).get(external_id)
        if not item:
            raise KeyError(f"Unknown storage object: {external_id}")
        item["status"] = status
        item["metadata"] = {**item.get("metadata", {}), **(metadata or {})}
        item["updated_at"] = utc_now()
        data.setdefault("events", []).append({
            "event_id": str(uuid.uuid4()),
            "external_id": external_id,
            "event_type": "status_changed",
            "actor": actor,
            "message": status,
            "payload": metadata or {},
            "created_at": utc_now(),
        })
        self._save(data)

    def delete_object(self, external_id: str) -> bool:
        data = self._load()
        existed = data.setdefault("objects", {}).pop(external_id, None) is not None
        data.setdefault("events", []).append({
            "event_id": str(uuid.uuid4()),
            "external_id": external_id,
            "event_type": "object_deleted",
            "actor": "storage",
            "message": "deleted" if existed else "missing",
            "payload": {},
            "created_at": utc_now(),
        })
        self._save(data)
        return existed

    def list_objects(self, object_type: str | None = None, status: str | None = None) -> list[dict]:
        objects = list(self._load().get("objects", {}).values())
        if object_type:
            objects = [item for item in objects if item.get("object_type") == object_type]
        if status:
            objects = [item for item in objects if item.get("status") == status]
        return sorted(objects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_object(self, external_id: str) -> dict | None:
        return self._load().get("objects", {}).get(external_id)

    def list_events(self, external_id: str | None = None, limit: int = 100) -> list[dict]:
        events = self._load().get("events", [])
        if external_id:
            events = [event for event in events if event.get("external_id") == external_id]
        return list(reversed(events[-limit:]))

    def metrics(self) -> dict:
        objects = self.list_objects()
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_size = 0
        for item in objects:
            by_type[item.get("object_type", "unknown")] = by_type.get(item.get("object_type", "unknown"), 0) + 1
            by_status[item.get("status", "unknown")] = by_status.get(item.get("status", "unknown"), 0) + 1
            total_size += int(item.get("size_bytes") or 0)
        return {
            "backend": "local",
            "objects": len(objects),
            "total_size_bytes": total_size,
            "by_type": by_type,
            "by_status": by_status,
            "events": len(self._load().get("events", [])),
        }

    def store_artifact(
        self,
        *,
        object_type: str,
        title: str,
        content: str,
        extension: str = "json",
        external_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        created_by_agent: str = "system",
    ) -> dict:
        external_id = external_id or str(uuid.uuid4())
        folder = self.artifact_dir / object_type
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{external_id}.{extension.lstrip('.')}"
        path.write_text(content, encoding="utf-8")
        content_hash = sha256_file(path)
        size_bytes = path.stat().st_size
        storage_uri = str(path)

        item = self.register_object(
            external_id=external_id,
            object_type=object_type,
            title=title,
            source_filename=path.name,
            mime_type=infer_mime_type(path.name),
            status="generated",
            storage_uri=storage_uri,
            content_hash=content_hash,
            size_bytes=size_bytes,
            tags=tags,
            metadata=metadata,
        )
        self.add_version(
            external_id=external_id,
            storage_uri=storage_uri,
            content_hash=content_hash,
            size_bytes=size_bytes,
            metadata=metadata,
            created_by_agent=created_by_agent,
        )
        return item

    def record_presentation(self, presentation: dict) -> dict:
        data = self._load()
        presentation_id = presentation["presentation_id"]
        data.setdefault("presentations", {})[presentation_id] = {
            **presentation,
            "created_at": presentation.get("created_at") or utc_now(),
        }
        data.setdefault("events", []).append({
            "event_id": str(uuid.uuid4()),
            "external_id": presentation.get("storage_external_id"),
            "event_type": "presentation_created",
            "actor": "voice_presenter",
            "message": presentation_id,
            "payload": {"topic": presentation.get("topic")},
            "created_at": utc_now(),
        })
        self._save(data)
        return data["presentations"][presentation_id]


class PostgresStorageHub(LocalStorageHub):
    def __init__(self, dsn: str, base_dir: str | Path = DEFAULT_STORAGE_DIR):
        super().__init__(base_dir=base_dir)
        self.dsn = dsn

    def _connect(self):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise RuntimeError(
                "psycopg2-binary is required for PostgreSQL storage. "
                "Install dependencies from requirements.txt."
            ) from exc
        return psycopg2.connect(self.dsn, cursor_factory=psycopg2.extras.RealDictCursor)

    def register_object(self, **kwargs) -> dict:
        import psycopg2.extras

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT adc_register_object(
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
                    ) AS object_id
                    """,
                    (
                        kwargs["external_id"],
                        kwargs["object_type"],
                        kwargs["title"],
                        kwargs.get("source_filename"),
                        kwargs.get("mime_type"),
                        kwargs.get("owner_session_id"),
                        kwargs.get("status", "stored"),
                        kwargs.get("storage_uri"),
                        kwargs.get("content_hash"),
                        kwargs.get("size_bytes"),
                        json_dumps(kwargs.get("tags") or []),
                        json_dumps(kwargs.get("metadata") or {}),
                    ),
                )
                object_id = str(cur.fetchone()["object_id"])
        return self.get_object(kwargs["external_id"]) or {"object_id": object_id, **kwargs}

    def add_version(self, **kwargs) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT adc_add_version(%s, %s, %s, %s, %s::jsonb, %s) AS version_id
                    """,
                    (
                        kwargs["external_id"],
                        kwargs.get("storage_uri"),
                        kwargs.get("content_hash"),
                        kwargs.get("size_bytes"),
                        json_dumps(kwargs.get("metadata") or {}),
                        kwargs.get("created_by_agent", "system"),
                    ),
                )
                version_id = str(cur.fetchone()["version_id"])
        return {"version_id": version_id, **kwargs}

    def set_status(self, external_id: str, status: str, metadata: dict | None = None, actor: str = "system") -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT adc_set_status(%s, %s, %s::jsonb, %s)",
                    (external_id, status, json_dumps(metadata or {}), actor),
                )

    def delete_object(self, external_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM adc_storage_objects WHERE external_id = %s", (external_id,))
                return cur.rowcount > 0

    def list_objects(self, object_type: str | None = None, status: str | None = None) -> list[dict]:
        where = []
        params: list[Any] = []
        if object_type:
            where.append("object_type = %s")
            params.append(object_type)
        if status:
            where.append("status = %s")
            params.append(status)
        clause = "WHERE " + " AND ".join(where) if where else ""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT external_id, object_id::text, object_type, title, source_filename,
                           mime_type, owner_session_id, status, storage_uri, content_hash,
                           size_bytes, tags, metadata, current_version_id::text,
                           created_at::text, updated_at::text
                      FROM adc_storage_objects
                      {clause}
                     ORDER BY updated_at DESC
                     LIMIT 500
                    """,
                    params,
                )
                return [dict(row) for row in cur.fetchall()]

    def get_object(self, external_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT external_id, object_id::text, object_type, title, source_filename,
                           mime_type, owner_session_id, status, storage_uri, content_hash,
                           size_bytes, tags, metadata, current_version_id::text,
                           created_at::text, updated_at::text
                      FROM adc_storage_objects
                     WHERE external_id = %s
                    """,
                    (external_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def list_events(self, external_id: str | None = None, limit: int = 100) -> list[dict]:
        params: list[Any] = []
        clause = ""
        if external_id:
            clause = "WHERE external_id = %s"
            params.append(external_id)
        params.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT event_id, object_id::text, external_id, event_type, actor,
                           message, payload, created_at::text
                      FROM adc_storage_events
                      {clause}
                     ORDER BY created_at DESC
                     LIMIT %s
                    """,
                    params,
                )
                return [dict(row) for row in cur.fetchall()]

    def metrics(self) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT object_type, status, COALESCE(size_bytes, 0) AS size_bytes FROM adc_storage_objects")
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) AS event_count FROM adc_storage_events")
                event_count = cur.fetchone()["event_count"]
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_size = 0
        for row in rows:
            by_type[row["object_type"]] = by_type.get(row["object_type"], 0) + 1
            by_status[row["status"]] = by_status.get(row["status"], 0) + 1
            total_size += int(row["size_bytes"] or 0)
        return {
            "backend": "postgres",
            "objects": len(rows),
            "total_size_bytes": total_size,
            "by_type": by_type,
            "by_status": by_status,
            "events": event_count,
        }

    def record_presentation(self, presentation: dict) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT adc_create_presentation(%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    AS presentation_id
                    """,
                    (
                        presentation.get("storage_external_id"),
                        presentation["topic"],
                        presentation.get("source_kind", "auto"),
                        presentation.get("source_ref"),
                        presentation.get("speech_text"),
                        json_dumps(presentation.get("voice_profile") or {}),
                        json_dumps(presentation.get("metadata") or {}),
                    ),
                )
                db_presentation_id = str(cur.fetchone()["presentation_id"])
                for index, segment in enumerate(presentation.get("segments", []), start=1):
                    cur.execute(
                        """
                        SELECT adc_add_presentation_segment(%s::uuid, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            db_presentation_id,
                            index,
                            segment.get("title", f"Segment {index}"),
                            segment.get("narration", ""),
                            segment.get("duration_hint_sec"),
                            json_dumps(segment.get("source_refs") or []),
                        ),
                    )
        presentation["database_presentation_id"] = db_presentation_id
        return presentation


def get_storage_hub():
    if STORAGE_BACKEND in {"postgres", "postgresql"}:
        if not POSTGRES_DSN:
            raise RuntimeError("POSTGRES_STORAGE_DSN or STORAGE_DATABASE_URL must be set for PostgreSQL storage.")
        return PostgresStorageHub(POSTGRES_DSN)
    return LocalStorageHub()


def sync_uploaded_file(
    *,
    file_id: str,
    filename: str,
    file_type: str,
    file_path: str,
    status: str,
    metadata: dict | None = None,
) -> dict | None:
    try:
        path = Path(file_path)
        return get_storage_hub().register_object(
            external_id=file_id,
            object_type="document",
            title=filename,
            source_filename=filename,
            mime_type=infer_mime_type(filename),
            status=status,
            storage_uri=str(path),
            content_hash=sha256_file(path) if path.exists() else None,
            size_bytes=path.stat().st_size if path.exists() else None,
            tags=[file_type],
            metadata=metadata or {},
        )
    except Exception as exc:
        logger.warning("StorageHub sync failed for file %s: %s", file_id, exc)
        return None
