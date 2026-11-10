-- Run with psql -X -v ON_ERROR_STOP=1 -f this-file.sql.
-- Deliberately restricted to the project's dedicated database.
\set ON_ERROR_STOP on
BEGIN;
DO $$ BEGIN
    IF current_database() <> 'live_feed_analysis' THEN
        RAISE EXCEPTION 'This migration requires the live_feed_analysis database';
    END IF;
END $$;
SELECT pg_advisory_xact_lock(16011987, 1);
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
SELECT NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE version = 1) AS apply_migration \gset
\if :apply_migration

CREATE TABLE public.sessions (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    ended_at timestamptz,
    status text NOT NULL DEFAULT 'starting'
        CHECK (status IN ('starting','running','stopping','completed','failed','interrupted')),
    question text NOT NULL CHECK (length(btrim(question)) BETWEEN 1 AND 2000),
    room_name text NOT NULL UNIQUE,
    publisher_identity text NOT NULL,
    worker_identity text,
    options jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(options) = 'object'),
    end_reason text,
    CHECK (started_at IS NULL OR started_at >= created_at),
    CHECK (ended_at IS NULL OR ended_at >= coalesce(started_at, created_at)),
    CHECK ((status IN ('completed','failed','interrupted')) = (ended_at IS NOT NULL))
);
CREATE INDEX sessions_recent ON public.sessions (created_at DESC);
CREATE INDEX sessions_status ON public.sessions (status, created_at DESC);
COMMENT ON COLUMN public.sessions.options IS 'Session settings only. Never store API keys, room JWTs or local access keys.';

CREATE TABLE public.vision_observations (
    session_id uuid NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    event_id uuid NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
    end_seconds double precision NOT NULL,
    state text NOT NULL CHECK (state IN ('response','standby','silence','error','snapshot')),
    description text,
    summary text,
    response text,
    sample_times double precision[] NOT NULL DEFAULT '{}',
    inference_seconds double precision NOT NULL DEFAULT 0
        CHECK (inference_seconds >= 0 AND inference_seconds < 'Infinity'::double precision),
    skipped_seconds double precision NOT NULL DEFAULT 0
        CHECK (skipped_seconds >= 0 AND skipped_seconds < 'Infinity'::double precision),
    image_budget integer NOT NULL DEFAULT 448 CHECK (image_budget IN (224,336,448)),
    input_width integer CHECK (input_width > 0),
    input_height integer CHECK (input_height > 0),
    evidence_status text NOT NULL DEFAULT 'not_requested',
    search_text tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, coalesce(description,'') || ' ' || coalesce(summary,'') || ' ' || coalesce(response,''))
    ) STORED,
    PRIMARY KEY (session_id, event_id),
    CHECK (end_seconds >= start_seconds AND end_seconds < 'Infinity'::double precision),
    CHECK (cardinality(sample_times) <= 4 AND array_position(sample_times, NULL) IS NULL
        AND start_seconds <= ALL(sample_times) AND end_seconds >= ALL(sample_times)),
    CHECK ((input_width IS NULL) = (input_height IS NULL))
);
CREATE INDEX vision_observations_time ON public.vision_observations (session_id, end_seconds DESC, start_seconds);
CREATE INDEX vision_observations_search ON public.vision_observations USING gin (search_text);
COMMENT ON TABLE public.vision_observations IS 'One row per inference batch, including silence. Silence does not establish that nothing happened.';
COMMENT ON COLUMN public.vision_observations.start_seconds IS 'GPU frame-reception clock relative to worker session creation; not browser capture time or the voice clock.';
COMMENT ON COLUMN public.vision_observations.sample_times IS 'Actual sampled frame times. Use these to derive coverage gaps; the batch envelope is not continuous video coverage.';

CREATE TABLE public.vision_evidence (
    session_id uuid NOT NULL,
    event_id uuid NOT NULL,
    frame_index smallint NOT NULL CHECK (frame_index IN (0,1)),
    sample_time_seconds double precision NOT NULL
        CHECK (sample_time_seconds >= 0 AND sample_time_seconds < 'Infinity'::double precision),
    created_at timestamptz NOT NULL DEFAULT now(),
    jpeg bytea NOT NULL CHECK (octet_length(jpeg) BETWEEN 1 AND 2000000),
    PRIMARY KEY (session_id,event_id,frame_index),
    FOREIGN KEY (session_id,event_id) REFERENCES public.vision_observations(session_id,event_id) ON DELETE CASCADE
);
CREATE INDEX vision_evidence_retention ON public.vision_evidence (created_at, session_id, event_id);
COMMENT ON TABLE public.vision_evidence IS 'Two cropped snapshots per retained inference batch. Evidence may be pruned independently of observation text; no continuous video recording.';

CREATE TABLE public.voice_sessions (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    provider_session_id text UNIQUE,
    model text NOT NULL,
    backend_model text,
    voice text,
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz,
    status text NOT NULL DEFAULT 'connecting'
        CHECK (status IN ('connecting','listening','closed','failed')),
    end_reason text,
    UNIQUE (session_id,id),
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK ((status IN ('closed','failed')) = (ended_at IS NOT NULL))
);
CREATE INDEX voice_sessions_camera ON public.voice_sessions (session_id, started_at);
COMMENT ON TABLE public.voice_sessions IS 'One row per voice connection, including reconnects. Each connection owns a separate audio timeline.';

CREATE TABLE public.voice_transcript_segments (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL,
    voice_session_id uuid NOT NULL,
    provider_event_id text NOT NULL,
    role text NOT NULL CHECK (role IN ('user','assistant')),
    text text NOT NULL,
    start_ms bigint NOT NULL CHECK (start_ms >= 0),
    end_ms bigint NOT NULL CHECK (end_ms >= start_ms),
    received_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (voice_session_id,provider_event_id),
    FOREIGN KEY (session_id,voice_session_id) REFERENCES public.voice_sessions(session_id,id) ON DELETE CASCADE
);
CREATE INDEX voice_transcript_timeline ON public.voice_transcript_segments (session_id,voice_session_id,start_ms,id);
COMMENT ON TABLE public.voice_transcript_segments IS 'Append-only transcript deltas, not assumed complete turns. Offsets use the corresponding voice connection clock, not GPU video time.';

CREATE TABLE public.model_calls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    voice_session_id uuid,
    observation_event_id uuid,
    component text NOT NULL CHECK (component IN ('onestreamer','voice','visual_review','memory_tool')),
    operation text NOT NULL,
    external_id text,
    model text,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','complete','error','cancelled','skipped')),
    input jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(input) = 'object'),
    output jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(output) = 'object'),
    error_code text,
    error_message text,
    UNIQUE (session_id,component,external_id),
    FOREIGN KEY (session_id,voice_session_id) REFERENCES public.voice_sessions(session_id,id) ON DELETE CASCADE,
    FOREIGN KEY (session_id,observation_event_id) REFERENCES public.vision_observations(session_id,event_id) ON DELETE CASCADE,
    CHECK (completed_at IS NULL OR completed_at >= started_at),
    CHECK ((status <> 'pending') = (completed_at IS NOT NULL))
);
CREATE INDEX model_calls_timeline ON public.model_calls (session_id,started_at,id);
CREATE INDEX model_calls_errors ON public.model_calls (session_id,started_at DESC) WHERE status='error';
COMMENT ON TABLE public.model_calls IS 'Inference/review/tool diagnostics. Input/output must be sanitized structured metadata; do not log credentials, raw audio or base64 images.';

CREATE TABLE public.debug_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    voice_session_id uuid,
    observation_event_id uuid,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    component text NOT NULL,
    level text NOT NULL CHECK (level IN ('debug','info','warning','error')),
    event_type text NOT NULL,
    message text NOT NULL,
    details jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(details) = 'object'),
    FOREIGN KEY (session_id,voice_session_id) REFERENCES public.voice_sessions(session_id,id) ON DELETE CASCADE,
    FOREIGN KEY (session_id,observation_event_id) REFERENCES public.vision_observations(session_id,event_id) ON DELETE CASCADE
);
CREATE INDEX debug_events_timeline ON public.debug_events (session_id,recorded_at,id);
CREATE INDEX debug_events_errors ON public.debug_events (session_id,recorded_at DESC) WHERE level IN ('warning','error');
COMMENT ON TABLE public.debug_events IS 'Lifecycle, reconnect, dropped frame/context, evidence transfer and error diagnostics. Sanitize provider errors and never log credentials.';

INSERT INTO public.schema_migrations(version,name) VALUES (1,'session_storage');
\endif
COMMIT;
