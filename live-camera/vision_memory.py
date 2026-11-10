"""Local, session-scoped visual timeline. Text outlives bounded JPEG evidence."""
import base64
import io
import json
import re
import sqlite3
import threading
from pathlib import Path

from protocol import MemoryQuery


class VisionMemory:
    def __init__(self, path, *, evidence_limit=256, byte_limit=128_000_000):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.evidence_limit, self.byte_limit = evidence_limit, byte_limit
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS observations (
                session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                start REAL NOT NULL, end REAL NOT NULL, state TEXT NOT NULL,
                description TEXT, summary TEXT, response TEXT, sample_times TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(session_id,event_id));
            CREATE INDEX IF NOT EXISTS memory_time ON observations(session_id,start,end);
            CREATE TABLE IF NOT EXISTS evidence (
                session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                timestamps TEXT NOT NULL, first BLOB NOT NULL, last BLOB NOT NULL,
                PRIMARY KEY(session_id,event_id));
        ''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(observations)')}
        if 'sample_times' not in columns:
            self.db.execute("ALTER TABLE observations ADD COLUMN sample_times TEXT NOT NULL DEFAULT '[]'")
        self.db.commit()

    def record(self, observation):
        # Duplicate data/evidence deliveries enrich one entry, never create another turn.
        with self.lock, self.db:
            cursor = self.db.execute('''INSERT OR IGNORE INTO observations (session_id,event_id,start,end,state,description,summary,response,sample_times) VALUES (?,?,?,?,?,?,?,?,?)''',
                (observation.session_id, observation.event_id, observation.start, observation.end,
                 observation.state, observation.description, observation.summary, observation.response, json.dumps(observation.sample_times)))
            return cursor.rowcount == 1

    def save_evidence(self, observation, images, timestamps):
        blobs = []
        for image in images:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=80)
            blobs.append(output.getvalue())
        with self.lock, self.db:
            self.db.execute('INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)',
                (observation.session_id, observation.event_id, json.dumps(timestamps), *blobs))
            # Bound globally, including across sessions. Text and missing-evidence status remain.
            rows = self.db.execute('''SELECT rowid, length(first)+length(last) AS size
                FROM evidence ORDER BY rowid DESC''').fetchall()
            used = 0
            for i, row in enumerate(rows):
                used += row['size']
                if i >= self.evidence_limit or used > self.byte_limit:
                    self.db.execute('DELETE FROM evidence WHERE rowid=?', (row['rowid'],))

    def search(self, query: MemoryQuery):
        clauses, params = ['o.session_id=?'], [query.session_id]
        if query.start is not None:
            clauses.append('o.end>=?'); params.append(query.start)
        if query.end is not None:
            clauses.append('o.start<=?'); params.append(query.end)
        # Parameterized lexical lookup; no SQL/FTS expression supplied by the model.
        terms = re.findall(r'\w+', query.query.lower())[:12]
        if terms:
            clauses.append('('+' OR '.join(
                "lower(coalesce(o.description,'')||' '||coalesce(o.summary,'')||' '||coalesce(o.response,'')) LIKE ? ESCAPE '\\'"
                for _ in terms)+')')
            params.extend('%'+t.replace('_', '\\_')+'%' for t in terms)
        with self.lock:
            rows = self.db.execute('''SELECT o.*, e.event_id IS NOT NULL AS evidence_available
                FROM observations o LEFT JOIN evidence e USING(session_id,event_id) WHERE '''+
                ' AND '.join(clauses)+' ORDER BY o.end DESC LIMIT ?', (*params, query.limit)).fetchall()
            coverage = self.coverage(query.session_id, query.start, query.end)
        return {'entries': [{**dict(r), 'sample_times': json.loads(r['sample_times'])} for r in reversed(rows)], **coverage,
                'clock': 'GPU frame reception seconds since worker session creation; not exact capture time',
                'limitations': 'Sampled observations can miss brief events. Silence means no description, not no activity. Evidence is two cropped snapshots per retained batch, not continuous video.'}

    def coverage(self, sid, start=None, end=None):
        with self.lock:
            rows = self.db.execute('''SELECT start,end,sample_times FROM observations
                WHERE session_id=? AND state NOT IN ('error','snapshot') ORDER BY start''', (sid,)).fetchall()
        intervals = []
        sampled = []
        for row in rows:
            times = json.loads(row['sample_times'])
            if times:
                sampled.extend((t, min(t+.25, row['end'])) for t in times)
            else:
                sampled.append((row['start'], row['end']))
        for lo, hi in sorted(sampled):
            if intervals and lo <= intervals[-1][1]+.001:
                intervals[-1][1] = max(intervals[-1][1], hi)
            else:
                intervals.append([lo, hi])
        lo = start if start is not None else (intervals[0][0] if intervals else 0)
        hi = end if end is not None else (intervals[-1][1] if intervals else lo)
        selected = [[max(a, lo), min(b, hi)] for a, b in intervals if b >= lo and a <= hi]
        gaps, cursor = [], lo
        for a, b in selected:
            if a > cursor + .001:
                gaps.append([cursor, a])
            cursor = max(cursor, b)
        if cursor < hi:
            gaps.append([cursor, hi])
        return {'coverage': selected[-100:], 'gaps': gaps[-100:],
                'latest_observation_end': intervals[-1][1] if intervals else None,
                'coverage_truncated': len(selected) > 100 or len(gaps) > 100}

    def inspect(self, query: MemoryQuery):
        result = self.search(query)
        content = [{'type': 'input_text', 'text': json.dumps(result)}]
        with self.lock:
            # At most six images, selected from across the returned interval.
            entries = [e for e in result['entries'] if e['evidence_available']]
            if len(entries) > 3:
                entries = [entries[0], entries[len(entries)//2], entries[-1]]
            for entry in entries:
                row = self.db.execute('SELECT * FROM evidence WHERE session_id=? AND event_id=?',
                    (query.session_id, entry['event_id'])).fetchone()
                if row is None:
                    continue
                for timestamp, blob in zip(json.loads(row['timestamps']), (row['first'], row['last'])):
                    content.append({'type': 'input_text', 'text': f"Evidence {entry['event_id']} at {timestamp:.3f}s"})
                    content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,'+
                                    base64.b64encode(blob).decode(), 'detail': 'auto'})
        return content

    def close(self):
        with self.lock:
            self.db.close()
