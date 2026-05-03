-- AI Data Copilot - PostgreSQL storage schema
-- Target: PostgreSQL 14+
-- Recommended database: ai_data_copilot

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS adc_storage_objects (
    object_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id         TEXT UNIQUE,
    object_type         TEXT NOT NULL,
    title               TEXT NOT NULL,
    source_filename     TEXT,
    mime_type           TEXT,
    owner_session_id    TEXT,
    status              TEXT NOT NULL DEFAULT 'stored',
    storage_uri         TEXT,
    content_hash        TEXT,
    size_bytes          BIGINT,
    tags                JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    current_version_id  UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS adc_storage_versions (
    version_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id           UUID NOT NULL REFERENCES adc_storage_objects(object_id) ON DELETE CASCADE,
    version_number      INTEGER NOT NULL,
    storage_uri         TEXT,
    content_hash        TEXT,
    size_bytes          BIGINT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by_agent    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (object_id, version_number)
);

CREATE TABLE IF NOT EXISTS adc_storage_events (
    event_id            BIGSERIAL PRIMARY KEY,
    object_id           UUID REFERENCES adc_storage_objects(object_id) ON DELETE SET NULL,
    external_id         TEXT,
    event_type          TEXT NOT NULL,
    actor               TEXT,
    message             TEXT,
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS adc_presentations (
    presentation_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    storage_object_id   UUID REFERENCES adc_storage_objects(object_id) ON DELETE SET NULL,
    topic               TEXT NOT NULL,
    source_kind         TEXT NOT NULL DEFAULT 'auto',
    source_ref          TEXT,
    status              TEXT NOT NULL DEFAULT 'generated',
    script              TEXT,
    voice_profile       JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS adc_presentation_segments (
    segment_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    presentation_id     UUID NOT NULL REFERENCES adc_presentations(presentation_id) ON DELETE CASCADE,
    position_number     INTEGER NOT NULL,
    title               TEXT NOT NULL,
    narration_text      TEXT NOT NULL,
    duration_hint_sec   INTEGER,
    source_refs         JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (presentation_id, position_number)
);

CREATE INDEX IF NOT EXISTS adc_storage_objects_type_ix
    ON adc_storage_objects(object_type);

CREATE INDEX IF NOT EXISTS adc_storage_objects_status_ix
    ON adc_storage_objects(status);

CREATE INDEX IF NOT EXISTS adc_storage_objects_metadata_gin
    ON adc_storage_objects USING gin(metadata);

CREATE INDEX IF NOT EXISTS adc_storage_objects_tags_gin
    ON adc_storage_objects USING gin(tags);

CREATE INDEX IF NOT EXISTS adc_storage_events_object_ix
    ON adc_storage_events(object_id, created_at DESC);

CREATE INDEX IF NOT EXISTS adc_storage_events_external_ix
    ON adc_storage_events(external_id, created_at DESC);

CREATE INDEX IF NOT EXISTS adc_presentations_created_ix
    ON adc_presentations(created_at DESC);

CREATE OR REPLACE FUNCTION adc_touch_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS adc_storage_objects_touch_trg ON adc_storage_objects;
CREATE TRIGGER adc_storage_objects_touch_trg
BEFORE UPDATE ON adc_storage_objects
FOR EACH ROW EXECUTE FUNCTION adc_touch_updated_at();

CREATE OR REPLACE FUNCTION adc_log_event(
    p_object_id UUID,
    p_external_id TEXT,
    p_event_type TEXT,
    p_actor TEXT DEFAULT 'system',
    p_message TEXT DEFAULT NULL,
    p_payload JSONB DEFAULT '{}'::jsonb
)
RETURNS BIGINT AS $$
DECLARE
    v_event_id BIGINT;
BEGIN
    INSERT INTO adc_storage_events (
        object_id, external_id, event_type, actor, message, payload
    ) VALUES (
        p_object_id, p_external_id, p_event_type, p_actor, p_message, COALESCE(p_payload, '{}'::jsonb)
    )
    RETURNING event_id INTO v_event_id;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION adc_register_object(
    p_external_id TEXT,
    p_object_type TEXT,
    p_title TEXT,
    p_source_filename TEXT DEFAULT NULL,
    p_mime_type TEXT DEFAULT NULL,
    p_owner_session_id TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'stored',
    p_storage_uri TEXT DEFAULT NULL,
    p_content_hash TEXT DEFAULT NULL,
    p_size_bytes BIGINT DEFAULT NULL,
    p_tags JSONB DEFAULT '[]'::jsonb,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    v_object_id UUID;
BEGIN
    INSERT INTO adc_storage_objects (
        external_id, object_type, title, source_filename, mime_type,
        owner_session_id, status, storage_uri, content_hash, size_bytes,
        tags, metadata
    ) VALUES (
        p_external_id, p_object_type, p_title, p_source_filename, p_mime_type,
        p_owner_session_id, COALESCE(p_status, 'stored'), p_storage_uri,
        p_content_hash, p_size_bytes, COALESCE(p_tags, '[]'::jsonb),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (external_id) DO UPDATE SET
        object_type = EXCLUDED.object_type,
        title = EXCLUDED.title,
        source_filename = EXCLUDED.source_filename,
        mime_type = EXCLUDED.mime_type,
        owner_session_id = EXCLUDED.owner_session_id,
        status = EXCLUDED.status,
        storage_uri = EXCLUDED.storage_uri,
        content_hash = EXCLUDED.content_hash,
        size_bytes = EXCLUDED.size_bytes,
        tags = EXCLUDED.tags,
        metadata = adc_storage_objects.metadata || EXCLUDED.metadata
    RETURNING object_id INTO v_object_id;

    PERFORM adc_log_event(
        v_object_id,
        p_external_id,
        'object_registered',
        'storage',
        COALESCE(p_status, 'stored'),
        COALESCE(p_metadata, '{}'::jsonb)
    );

    RETURN v_object_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION adc_add_version(
    p_external_id TEXT,
    p_storage_uri TEXT DEFAULT NULL,
    p_content_hash TEXT DEFAULT NULL,
    p_size_bytes BIGINT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_created_by_agent TEXT DEFAULT 'system'
)
RETURNS UUID AS $$
DECLARE
    v_object_id UUID;
    v_version_id UUID;
    v_version_number INTEGER;
BEGIN
    SELECT object_id INTO v_object_id
      FROM adc_storage_objects
     WHERE external_id = p_external_id;

    IF v_object_id IS NULL THEN
        RAISE EXCEPTION 'Unknown storage external_id: %', p_external_id;
    END IF;

    SELECT COALESCE(MAX(version_number), 0) + 1
      INTO v_version_number
      FROM adc_storage_versions
     WHERE object_id = v_object_id;

    INSERT INTO adc_storage_versions (
        object_id, version_number, storage_uri, content_hash, size_bytes,
        metadata, created_by_agent
    ) VALUES (
        v_object_id, v_version_number, p_storage_uri, p_content_hash, p_size_bytes,
        COALESCE(p_metadata, '{}'::jsonb), p_created_by_agent
    )
    RETURNING version_id INTO v_version_id;

    UPDATE adc_storage_objects
       SET current_version_id = v_version_id,
           storage_uri = COALESCE(p_storage_uri, storage_uri),
           content_hash = COALESCE(p_content_hash, content_hash),
           size_bytes = COALESCE(p_size_bytes, size_bytes),
           metadata = metadata || COALESCE(p_metadata, '{}'::jsonb)
     WHERE object_id = v_object_id;

    PERFORM adc_log_event(
        v_object_id,
        p_external_id,
        'version_added',
        COALESCE(p_created_by_agent, 'system'),
        v_version_id::text,
        COALESCE(p_metadata, '{}'::jsonb)
    );

    RETURN v_version_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION adc_set_status(
    p_external_id TEXT,
    p_status TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_actor TEXT DEFAULT 'system'
)
RETURNS VOID AS $$
DECLARE
    v_object_id UUID;
BEGIN
    UPDATE adc_storage_objects
       SET status = p_status,
           metadata = metadata || COALESCE(p_metadata, '{}'::jsonb)
     WHERE external_id = p_external_id
     RETURNING object_id INTO v_object_id;

    IF v_object_id IS NULL THEN
        RAISE EXCEPTION 'Unknown storage external_id: %', p_external_id;
    END IF;

    PERFORM adc_log_event(
        v_object_id,
        p_external_id,
        'status_changed',
        COALESCE(p_actor, 'system'),
        p_status,
        COALESCE(p_metadata, '{}'::jsonb)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION adc_create_presentation(
    p_external_id TEXT,
    p_topic TEXT,
    p_source_kind TEXT DEFAULT 'auto',
    p_source_ref TEXT DEFAULT NULL,
    p_script TEXT DEFAULT NULL,
    p_voice_profile JSONB DEFAULT '{}'::jsonb,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    v_storage_object_id UUID;
    v_presentation_id UUID;
BEGIN
    SELECT object_id INTO v_storage_object_id
      FROM adc_storage_objects
     WHERE external_id = p_external_id;

    INSERT INTO adc_presentations (
        storage_object_id, topic, source_kind, source_ref, script,
        voice_profile, metadata
    ) VALUES (
        v_storage_object_id, p_topic, COALESCE(p_source_kind, 'auto'), p_source_ref,
        p_script, COALESCE(p_voice_profile, '{}'::jsonb),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING presentation_id INTO v_presentation_id;

    PERFORM adc_log_event(
        v_storage_object_id,
        p_external_id,
        'presentation_created',
        'voice_presenter',
        v_presentation_id::text,
        COALESCE(p_metadata, '{}'::jsonb)
    );

    RETURN v_presentation_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION adc_add_presentation_segment(
    p_presentation_id UUID,
    p_position_number INTEGER,
    p_title TEXT,
    p_narration_text TEXT,
    p_duration_hint_sec INTEGER DEFAULT NULL,
    p_source_refs JSONB DEFAULT '[]'::jsonb
)
RETURNS UUID AS $$
DECLARE
    v_segment_id UUID;
BEGIN
    INSERT INTO adc_presentation_segments (
        presentation_id, position_number, title, narration_text,
        duration_hint_sec, source_refs
    ) VALUES (
        p_presentation_id, p_position_number, p_title, p_narration_text,
        p_duration_hint_sec, COALESCE(p_source_refs, '[]'::jsonb)
    )
    RETURNING segment_id INTO v_segment_id;

    RETURN v_segment_id;
END;
$$ LANGUAGE plpgsql;
