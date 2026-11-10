# Postgres session storage

The dedicated database is `live_feed_analysis`. Migration `001_session_storage.sql`
creates the initial schema atomically and records its version in `schema_migrations`.
Running the migration again skips its completed version. It refuses any other database
and uses a transaction-scoped advisory lock to serialize concurrent migration runs.
Apply future changes in new numbered migrations; do not edit a migration already applied.

```sh
psql "$DATABASE_URL" -X -v ON_ERROR_STOP=1 -f live-camera/migrations/001_session_storage.sql
psql "$DATABASE_URL" -X -v ON_ERROR_STOP=1 -f live-camera/test_database.sql
```

`DATABASE_URL` is a standard Postgres URL, for example
`postgresql://mridulrao@localhost:5432/live_feed_analysis`. The commands above expect it
in the shell environment; psql does not load `.env` automatically. Database tests use
transactional fixtures and roll them back; identity sequences may advance.

| Table | Responsibility |
|---|---|
| `sessions` | Camera session lifecycle, original instruction, room and non-secret settings |
| `vision_observations` | Every inference interval, actual sample times, descriptions, summaries, responses, timings and evidence transport status |
| `vision_evidence` | Up to two cropped JPEG snapshots per retained observation |
| `voice_sessions` | Each voice connection, provider session ID, model/voice and lifecycle |
| `voice_transcript_segments` | Append-only user/assistant transcript deltas with provider event deduplication |
| `model_calls` | Inference, visual review and memory tool requests/results, errors and execution times |
| `debug_events` | Session lifecycle, dropped updates, tracking/transport diagnostics and errors |

All application rows belong to a session. Composite foreign keys also prevent evidence,
voice transcripts, model calls and debug links from referring to another session's data.
Deleting a session cascades to its children. Deleting retained snapshots leaves the
observation text intact. Evidence availability is derived from rows in `vision_evidence`;
`evidence_status` records transport status, which is separate from retention.

Video times are seconds relative to GPU worker session creation; voice offsets are
milliseconds relative to a specific voice connection. `received_at` and lifecycle fields
are UTC-capable `timestamptz`. Neither offsets nor arrival timestamps establish exact
capture/audio synchronization. Silence observations do not prove that nothing happened.

Use `model_calls.input` / `output` for sanitized structured details such as review
assessments, tool arguments/results, model usage and external request IDs. Use
`debug_events.details` for extensible diagnostics. Do not store credentials, room tokens,
raw audio or base64 media in either. Raw video is not stored by this schema. Snapshot
retention must be implemented by the application; tables do not schedule automatic pruning.

The schema is created independently of the runtime. The current backend still uses
SQLite vision memory and JSONL logs; setting `DATABASE_URL` alone does not switch storage.
Runtime writes, lifecycle recovery and any existing-data migration need a separate change.

## Useful queries

```sql
-- Most recent sessions.
SELECT id, created_at, status, question, ended_at, end_reason
FROM sessions ORDER BY created_at DESC LIMIT 20;

-- Timeline for one session (replace the UUID).
SELECT start_seconds, end_seconds, state, description, summary, response,
       inference_seconds, evidence_status
FROM vision_observations
WHERE session_id = '00000000-0000-0000-0000-000000000000'
ORDER BY start_seconds;

-- Search descriptions within a single session.
SELECT event_id, start_seconds, end_seconds, description, summary, response
FROM vision_observations
WHERE session_id = '00000000-0000-0000-0000-000000000000'
  AND search_text @@ plainto_tsquery('simple', 'cup laptop')
ORDER BY end_seconds DESC LIMIT 20;

-- Warning/error diagnostics for one session.
SELECT recorded_at, component, event_type, message, details
FROM debug_events
WHERE session_id = '00000000-0000-0000-0000-000000000000'
  AND level IN ('warning', 'error')
ORDER BY recorded_at DESC;
```
