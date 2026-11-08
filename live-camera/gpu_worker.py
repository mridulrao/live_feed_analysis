"""Outbound-only LiveKit worker. No HTTP listener and no OpenAI dependency."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import json
import logging
import signal
import time
import uuid

from livekit import rtc
from PIL import Image

from inference import OneStreamer
from protocol import (EVENT_TOPIC, EVIDENCE_TOPIC, Observation, SessionID,
                      WorkerStart, crop_frame, encode_evidence)
from settings import Settings

log = logging.getLogger('gpu-worker')


class GPUWorker:
    def __init__(self, settings, engine_factory=OneStreamer):
        self.settings, self.engine_factory = settings, engine_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='onestreamer')
        self.engine = None
        self.control = rtc.Room()
        self.active = None
        self.guard = asyncio.Lock()

    async def gpu(self, fn, *args):
        # Shield prevents cancellation from releasing ownership of running CUDA work.
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(self.executor, fn, *args))

    async def connect(self):
        if self.engine is None:
            self.engine = await self.gpu(self.engine_factory)
        await self.control.connect(self.settings.url,
            self.settings.token(self.settings.worker_identity, self.settings.control_room),
            rtc.RoomOptions(auto_subscribe=False))
        self.control.local_participant.register_rpc_method('camera.control', self.rpc)
        log.info('Model ready; waiting in control room %s', self.settings.control_room)

    async def rpc(self, invocation):
        if invocation.caller_identity != self.settings.backend_identity:
            return json.dumps({'error': 'Unauthorized coordinator'})
        try:
            body = json.loads(invocation.payload)
            action = body.pop('action')
            async with self.guard:
                if action == 'start':
                    options = WorkerStart.model_validate(body)
                    if self.active:
                        raise ValueError('GPU is busy or finishing the previous session')
                    session = VideoSession(self, options)
                    self.active = session
                    try:
                        await session.open()
                    except BaseException:
                        await session.close()
                        self.active = None
                        raise
                    session.task = asyncio.create_task(self.run_session(session))
                    return json.dumps({'ready': True})
                sid = SessionID.model_validate(body).session_id
                if action == 'stop' and self.active is None:
                    return json.dumps({'ok': True})
                if not self.active or self.active.options.session_id != sid:
                    raise ValueError('Session ended')
                if action == 'heartbeat':
                    self.active.touched = time.monotonic()
                elif action == 'stop':
                    self.active.stopped.set()
                elif action == 'verify':
                    return json.dumps(await self.active.manual_review())
                else:
                    raise ValueError('Unknown action')
                return json.dumps({'ok': True})
        except Exception as exc:
            log.warning('Control request failed: %s', type(exc).__name__)
            return json.dumps({'error': str(exc) if isinstance(exc, ValueError) else 'GPU command failed'})

    async def run_session(self, session):
        try:
            await session.run()
        except Exception:
            log.exception('Session failed')
            with suppress(Exception):
                await session.publish(Observation(
                    session_id=session.options.session_id, event_id=uuid.uuid4().hex,
                    state='error', start=0, end=0,
                    response='GPU session failed. Stop and start a new session.'))
        finally:
            try:
                await session.close()
            finally:
                if self.active is session:
                    self.active = None

    async def close(self):
        if self.active:
            self.active.stopped.set()
            if self.active.task:
                await self.active.task
        await self.control.disconnect()
        self.executor.shutdown(wait=True, cancel_futures=True)


class VideoSession:
    def __init__(self, worker, options):
        self.worker, self.options = worker, options
        self.room = rtc.Room()
        self.frames = deque(maxlen=4)
        self.stopped = asyncio.Event()
        self.epoch = self.touched = time.monotonic()
        self.last_sample = float('-inf')
        self.last_consumed = -1
        self.receiver = self.sender = self.task = None
        self.evidence = asyncio.Queue(maxsize=1)
        self.sending = False

    async def open(self):
        @self.room.on('track_published')
        def published(publication, participant):
            if (participant.identity == self.options.publisher
                    and publication.source == rtc.TrackSource.SOURCE_CAMERA):
                publication.set_subscribed(True)

        @self.room.on('track_subscribed')
        def subscribed(track, publication, participant):
            if (participant.identity == self.options.publisher
                    and publication.source == rtc.TrackSource.SOURCE_CAMERA
                    and track.kind == rtc.TrackKind.KIND_VIDEO and self.receiver is None):
                self.receiver = asyncio.create_task(self.receive(track))

        @self.room.on('participant_disconnected')
        def departed(participant):
            if participant.identity in (self.options.publisher, self.worker.settings.backend_identity):
                self.stopped.set()

        @self.room.on('track_unsubscribed')
        def unsubscribed(track, publication, participant):
            if participant.identity == self.options.publisher:
                self.stopped.set()

        @self.room.on('disconnected')
        def disconnected(*_):
            self.stopped.set()

        await self.worker.gpu(self.worker.engine.start, self.options)
        await self.room.connect(self.worker.settings.url,
            self.worker.settings.token(self.worker.settings.worker_identity, self.options.room),
            rtc.RoomOptions(auto_subscribe=False))
        self.sender = asyncio.create_task(self.send_evidence())

    async def receive(self, track):
        stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGB24)
        try:
            async for frame_event in stream:
                now = time.monotonic()
                if self.stopped.is_set():
                    break
                if now-self.last_sample < .25:
                    continue
                self.last_sample = now
                frame = frame_event.frame
                if min(frame.width, frame.height) < 32 or max(frame.width, frame.height) > 4096:
                    raise ValueError('Unexpected video dimensions')
                image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.data))
                rotation = {rtc.VideoRotation.VIDEO_ROTATION_90: 90,
                            rtc.VideoRotation.VIDEO_ROTATION_180: 180,
                            rtc.VideoRotation.VIDEO_ROTATION_270: 270}.get(frame_event.rotation, 0)
                if rotation:
                    image = image.rotate(-rotation, expand=True)
                image.thumbnail((1280, 1280))
                # Receiver-relative time: do not misrepresent arrival time as capture time.
                self.frames.append((now-self.epoch, image))
        except Exception:
            log.exception('Video receiver failed')
        finally:
            await stream.aclose()
            self.stopped.set()

    async def publish(self, observation):
        await asyncio.wait_for(self.room.local_participant.publish_data(
            observation.model_dump_json().encode(), reliable=True, topic=EVENT_TOPIC,
            destination_identities=[self.worker.settings.backend_identity]), 5)

    def queue_evidence(self, observation, images, times):
        if self.sending or not self.evidence.empty():
            observation.evidence_status = 'busy'
            return False
        observation.evidence_status = 'pending'
        self.evidence.put_nowait((observation, images, times))
        return True

    async def send_evidence(self):
        while True:
            observation, images, times = await self.evidence.get()
            self.sending = True
            try:
                packet = await asyncio.to_thread(encode_evidence, observation, images, times)
                await asyncio.wait_for(self.room.local_participant.send_bytes(
                    packet, topic=EVIDENCE_TOPIC, name=observation.event_id,
                    mime_type='application/octet-stream',
                    destination_identities=[self.worker.settings.backend_identity]), 15)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception('Evidence transfer failed')
                observation.evidence_status = 'transfer_failed'
                with suppress(Exception):
                    await self.publish(observation)
            finally:
                self.sending = False
                self.evidence.task_done()

    async def manual_review(self):
        if not self.options.verify:
            raise ValueError('Enable OpenAI verification when starting the session')
        if len(self.frames) < 4 or time.monotonic()-self.epoch-self.frames[-1][0] > 2:
            raise ValueError('Waiting for fresh camera frames')
        batch = list(self.frames)
        times = [t for t, _ in batch]
        observation = Observation(session_id=self.options.session_id, event_id=uuid.uuid4().hex,
            state='snapshot', start=times[0], end=times[-1]+.25)
        images = [crop_frame(im, self.options.crop) for _, im in batch]
        queued = self.queue_evidence(observation, images, times)
        await self.publish(observation)
        return {'event_id': observation.event_id, 'status': 'pending' if queued else 'busy'}

    async def run(self):
        while not self.stopped.is_set():
            if time.monotonic()-self.touched > 45:
                break
            if len(self.frames) == 4 and self.frames[0][0] > self.last_consumed:
                batch = list(self.frames)
                self.last_consumed = batch[-1][0]
                result, images, times = await self.worker.gpu(self.worker.engine.step, batch)
                if self.stopped.is_set():
                    break
                if self.options.verify and result.state == 'response':
                    self.queue_evidence(result, images, times)
                await self.publish(result)
            await asyncio.sleep(.05)

    async def close(self):
        self.stopped.set()
        for task in (self.receiver, self.sender):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        try:
            await self.room.disconnect()
        finally:
            await self.worker.gpu(self.worker.engine.stop)
            self.frames.clear()


async def main():
    settings = Settings()
    if not settings.configured:
        raise RuntimeError(settings.configuration_error)
    worker, done = GPUWorker(settings), asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, done.set)
    try:
        # Load failures should fail startup; transient LiveKit outages should retry
        # without repeatedly loading another model into GPU memory.
        worker.engine = await worker.gpu(worker.engine_factory)
        delay = 1
        while not done.is_set():
            if not worker.control.isconnected():
                if worker.active:
                    worker.active.stopped.set()
                try:
                    await worker.connect()
                    delay = 1
                except Exception as exc:
                    log.warning('Control reconnect unavailable (%s)', type(exc).__name__)
                    delay = min(delay*2, 30)
            try:
                await asyncio.wait_for(done.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
    finally:
        await worker.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
