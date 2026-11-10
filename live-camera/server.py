"""Local coordinator/UI: LiveKit orchestration and optional OpenAI evidence review."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from livekit import api, rtc

from protocol import (EVENT_TOPIC, EVIDENCE_TOPIC, MAX_EVIDENCE_BYTES,
                      Observation, SessionID, Start, WorkerStart, decode_evidence, TRACK_TOPIC, TrackingUpdate, TrackingControl)
from protocol import MemoryQuery
from settings import ROOT, Settings
from verification import Reviewer
from vision_memory import VisionMemory
from voice import VoiceAgent

log = logging.getLogger('camera-backend')


class CameraSession:
    def __init__(self, options, output=None, memory=None):
        self.options = options
        self.output = output
        self.room = rtc.Room()
        self.events = OrderedDict()
        self.tracking = None
        self.tracking_received = 0
        self.tracking_generation = 0
        self.evidence_seen = set()
        self.evidence_failures = OrderedDict()
        self.evidence_task = None
        self.touched = time.monotonic()
        self.active = True
        self.memory = memory
        self.voice = None
        self.last_visual_received = 0

    def record(self, observation):
        if self.memory and self.options.vision_memory:
            fresh = self.memory.record(observation)
            if fresh and observation.state not in ('error', 'snapshot'):
                self.last_visual_received = time.monotonic()
                if self.voice:
                    self.voice.observe(observation)
        row = observation.model_dump()
        if observation.event_id in self.evidence_failures:
            row['evidence_status'] = 'transfer_failed'
        old = self.events.get(observation.event_id, {})
        if old.get('evidence_status') == 'received':
            row['evidence_status'] = 'received'
        if 'review' in old:
            row['review'] = old['review']
            row['evidence_status'] = old['evidence_status']
        self.events[observation.event_id] = row
        while len(self.events) > 200:
            event_id, _ = self.events.popitem(last=False)
            self.evidence_seen.discard(event_id)
        if self.output:
            try:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                with self.output.open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
            except OSError:
                log.warning('Could not write observation log')
        return row

    def evidence_failed(self, event_id):
        if not isinstance(event_id, str) or not re.fullmatch(r'[a-f0-9]{32}', event_id):
            return
        self.evidence_failures[event_id] = True
        while len(self.evidence_failures) > 200:
            self.evidence_failures.popitem(last=False)
        if event_id in self.events and 'review' not in self.events[event_id]:
            self.events[event_id]['evidence_status'] = 'transfer_failed'


class Coordinator:
    def __init__(self, settings=None, reviewer=None, workspace=None):
        self.settings = settings or Settings()
        self.workspace = Path(workspace or os.environ.get('LOCAL_WORKSPACE', ROOT.parent / '.runtime'))
        self.workspace.mkdir(parents=True, exist_ok=True)
        key_path = self.workspace / 'local-access-key'
        configured_key = os.environ.get('LOCAL_ACCESS_KEY', '')
        if not configured_key and not key_path.exists():
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(secrets.token_urlsafe(32))
        self.access_key = configured_key or key_path.read_text().strip()
        if len(self.access_key) < 24:
            raise ValueError('LOCAL_ACCESS_KEY must be at least 24 characters')
        self.reviewer = reviewer if reviewer is not None else Reviewer(self.workspace)
        self.memory = VisionMemory(self.workspace / 'vision-memory.sqlite3')
        self.control = rtc.Room()
        self.session = None
        self.guard = asyncio.Lock()
        self.tasks = []

    @property
    def worker_ready(self):
        return (self.control.isconnected()
                and self.settings.worker_identity in self.control.remote_participants)

    async def connect(self):
        # The local page stays available when LiveKit is unconfigured/offline.
        if self.settings.configured:
            self.tasks.append(asyncio.create_task(self.connect_control()))
        self.tasks.append(asyncio.create_task(self.maintain()))

    async def connect_control(self):
        delay = 1
        while True:
            if not self.control.isconnected():
                try:
                    await self.control.connect(self.settings.url,
                        self.settings.token(self.settings.backend_identity, self.settings.control_room),
                        rtc.RoomOptions(auto_subscribe=False))
                    delay = 1
                except Exception as exc:
                    log.warning('LiveKit connection unavailable (%s)', type(exc).__name__)
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 30)
                    continue
            await asyncio.sleep(2)

    async def command(self, action, options):
        if not self.worker_ready:
            raise HTTPException(503, 'GPU worker is not connected to LiveKit')
        body = options.model_dump() if hasattr(options, 'model_dump') else options
        try:
            raw = await self.control.local_participant.perform_rpc(
                destination_identity=self.settings.worker_identity, method='camera.control',
                payload=json.dumps({'action': action, **body}), response_timeout=25)
            response = json.loads(raw)
        except Exception as exc:
            raise HTTPException(503, 'GPU command did not complete; check worker logs') from exc
        if response.get('error'):
            raise HTTPException(409, response['error'])
        return response

    async def start(self, options):
        async with self.guard:
            if self.session and self.session.active:
                raise HTTPException(409, 'A camera session is already active')
            if not self.worker_ready:
                raise HTTPException(503, 'GPU worker is not connected to LiveKit')
            if options.verify and not self.reviewer.configured:
                raise HTTPException(422, 'Configure OPENAI_API_KEY in the local backend')
            if options.voice and not os.environ.get('OPENAI_API_KEY', ''):
                raise HTTPException(422, 'Configure OPENAI_API_KEY in the local backend for GPT-Live')
            sid = uuid.uuid4().hex
            worker_options = WorkerStart(**options.model_dump(), session_id=sid,
                room='camera-'+sid, publisher='camera-'+sid)
            session = CameraSession(worker_options, self.workspace / 'outputs' / f'camera-{sid}.jsonl', self.memory)
            self.session = session
            self.attach_handlers(session)
            try:
                await session.room.connect(self.settings.url,
                    self.settings.token(self.settings.backend_identity, worker_options.room),
                    rtc.RoomOptions(auto_subscribe=False))
                await self.command('start', worker_options)
                if options.voice:
                    session.voice = VoiceAgent(self.settings, session, self.memory)
                    try:
                        await session.voice.open()
                    except Exception as exc:
                        raise HTTPException(503, 'GPT-Live could not start. Check OPENAI_API_KEY, model access and backend logs.') from exc
            except BaseException:
                # On an RPC timeout the worker may already have started. Disconnecting
                # this room also tells that worker to finish, with its lease as fallback.
                session.active = False
                if session.voice:
                    await session.voice.close()
                with suppress(HTTPException):
                    await self.command('stop', {'session_id': sid})
                await session.room.disconnect()
                raise
            return dict(session_id=sid, url=self.settings.url,
                token=self.settings.token(worker_options.publisher, worker_options.room, camera=True, voice=options.voice),
                fps=4, window_frames=32, image_budget=options.image_budget,
                verification_enabled=options.verify, voice_enabled=options.voice,
                voice_identity=session.voice.identity if session.voice else None,
                vision_memory_enabled=options.vision_memory)

    def attach_handlers(self, session):
        @session.room.on('data_received')
        def received(packet):
            if (packet.topic not in (EVENT_TOPIC, TRACK_TOPIC) or not packet.participant
                    or packet.participant.identity != self.settings.worker_identity
                    or not session.active or len(packet.data) > 15_000):
                return
            try:
                if packet.topic == TRACK_TOPIC:
                    update = TrackingUpdate.model_validate_json(packet.data)
                    if (update.session_id == session.options.session_id and session.options.tracking
                            and update.generation == session.tracking_generation
                            and (session.tracking is None or update.sequence > session.tracking['sequence'])):
                        session.tracking = update.model_dump()
                        session.tracking_received = time.monotonic()
                    return
                observation = Observation.model_validate_json(packet.data)
                if observation.session_id != session.options.session_id:
                    return
                session.record(observation)
            except ValueError:
                log.warning('Discarded invalid GPU event')

        def evidence(reader, participant_identity):
            if participant_identity != self.settings.worker_identity or not session.active:
                reader.close()
                return
            if session.evidence_task and not session.evidence_task.done():
                session.evidence_failed(reader.info.name)
                reader.close()
                return
            session.evidence_task = asyncio.create_task(self.receive_evidence(session, reader))
        session.room.register_byte_stream_handler(EVIDENCE_TOPIC, evidence)

    async def receive_evidence(self, session, reader):
        event_id = getattr(getattr(reader, 'info', None), 'name', None)
        try:
            data = bytearray()
            async with asyncio.timeout(20):
                async for chunk in reader:
                    if len(data)+len(chunk) > MAX_EVIDENCE_BYTES:
                        raise ValueError('Evidence byte limit exceeded')
                    data.extend(chunk)
            observation, images, timestamps = await asyncio.to_thread(decode_evidence, bytes(data))
            if event_id is not None and event_id != observation.event_id:
                raise ValueError('Evidence stream name does not match event ID')
            event_id = observation.event_id
            if (not session.active or self.session is not session
                    or observation.session_id != session.options.session_id
                    or not (session.options.verify or session.options.vision_memory)
                    or event_id in session.evidence_seen):
                return
            session.evidence_seen.add(event_id)
            row = session.record(observation)
            row['evidence_status'] = 'received'
            if session.options.vision_memory:
                await asyncio.to_thread(self.memory.save_evidence, observation, images, timestamps)
            if session.options.verify and observation.state in ('response', 'snapshot'):
                row['review'] = self.reviewer.enqueue(session.options.session_id,
                    session.options.question, observation.response or '', images, timestamps,
                    event_id=event_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('Evidence unavailable (%s)', type(exc).__name__)
            session.evidence_failed(event_id)
        finally:
            reader.close()

    def current(self, sid):
        if not self.session or self.session.options.session_id != sid:
            raise HTTPException(409, 'Unknown session')
        return self.session

    async def set_tracking(self, body):
        async with self.guard:
            session = self.current(body.session_id)
            if not session.active:
                raise HTTPException(409, 'No active session')
            result = await self.command('tracking', body)
            session.options.tracking = result['enabled']
            session.tracking_generation = result['generation']
            session.tracking = None
            session.tracking_received = 0
            return result

    async def stop(self, sid):
        async with self.guard:
            session = self.current(sid)
            if not session.active and not session.room.isconnected():
                return {'stopped': True}
            session.active = False
            if session.voice:
                await session.voice.close()
            if session.evidence_task:
                session.evidence_task.cancel()
                with suppress(asyncio.CancelledError):
                    await session.evidence_task
            with suppress(HTTPException):
                await self.command('stop', {'session_id': sid})
            await session.room.disconnect()
            # Remove the browser and end media publication even if its tab is stalled.
            if self.settings.configured:
                try:
                    async with api.LiveKitAPI(self.settings.url, self.settings.api_key,
                                              self.settings.api_secret) as client:
                        await client.room.delete_room(api.DeleteRoomRequest(room=session.options.room))
                except Exception as exc:
                    log.warning('Room cleanup unavailable (%s)', type(exc).__name__)
            return {'stopped': True}

    async def maintain(self):
        while True:
            await asyncio.sleep(10)
            session = self.session
            if not session or not session.active:
                continue
            sid = session.options.session_id
            if time.monotonic()-session.touched > 35:
                await self.stop(sid)
                continue
            try:
                await self.command('heartbeat', {'session_id': sid})
            except HTTPException:
                if self.session is not session or not session.active:
                    continue
                observation = Observation(session_id=sid, event_id=uuid.uuid4().hex,
                    state='error', start=0, end=0, response='GPU connection lost. Restart the session.')
                session.record(observation)
                await self.stop(sid)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task
        if self.session:
            await self.stop(self.session.options.session_id)
        await self.control.disconnect()
        await asyncio.to_thread(self.reviewer.close)
        await asyncio.to_thread(self.memory.close)


def create_app(coordinator=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.coordinator = coordinator or Coordinator()
        await app.state.coordinator.connect()
        try:
            yield
        finally:
            await app.state.coordinator.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def service():
        return app.state.coordinator

    def authorize(x_access_key: str = Header(default='')):
        if not secrets.compare_digest(x_access_key, service().access_key):
            raise HTTPException(401, 'Invalid local access key')

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'static/index.html', headers={'Cache-Control': 'no-store'})

    @app.get('/app.js')
    def javascript():
        return FileResponse(ROOT / 'static/app.js', media_type='text/javascript')

    @app.get('/vendor/livekit-client.umd.js')
    def livekit_javascript():
        path = ROOT / 'web/node_modules/livekit-client/dist/livekit-client.umd.js'
        if not path.exists():
            raise HTTPException(503, 'Run npm ci --prefix live-camera/web')
        return FileResponse(path, media_type='text/javascript')

    @app.get('/health')
    def health():
        c = service()
        return dict(ready=c.worker_ready, livekit_configured=c.settings.configured,
                    configuration_error=c.settings.configuration_error,
                    verification_configured=c.reviewer.configured, verification_model=c.reviewer.model,
                    voice_configured=bool(os.environ.get('OPENAI_API_KEY', '')),
                    voice_model=os.environ.get('OPENAI_LIVE_MODEL', 'gpt-live-1'))

    @app.post('/api/start', dependencies=[Depends(authorize)])
    async def start(body: Start):
        return await service().start(body)

    @app.post('/api/events', dependencies=[Depends(authorize)])
    def events(body: SessionID):
        c = service()
        session = c.current(body.session_id)
        session.touched = time.monotonic()
        return {'active': session.active, 'events': list(session.events.values()),
                'voice': session.voice.status() if session.voice else {'state': 'off'},
                'reviews': c.reviewer.get_results(body.session_id)}

    @app.post('/api/memory', dependencies=[Depends(authorize)])
    async def memory(body: MemoryQuery):
        c = service()
        # Explicit session IDs allow local lookup after restart; never cross-session search.
        return await asyncio.to_thread(c.memory.search, body)

    @app.post('/api/tracking', dependencies=[Depends(authorize)])
    async def tracking(body: TrackingControl):
        return await service().set_tracking(body)

    @app.post('/api/tracks', dependencies=[Depends(authorize)])
    def tracks(body: SessionID):
        session = service().current(body.session_id)
        update = session.tracking if session.active and session.options.tracking and time.monotonic()-session.tracking_received < 1 else None
        return {'tracking': update}

    @app.post('/api/verify', dependencies=[Depends(authorize)])
    async def verify(body: SessionID):
        c = service()
        session = c.current(body.session_id)
        if not session.active or not session.options.verify:
            raise HTTPException(409, 'No active session with verification enabled')
        return await c.command('verify', body)

    @app.post('/api/stop', dependencies=[Depends(authorize)])
    async def stop(body: SessionID):
        return await service().stop(body.session_id)

    return app


app = create_app()
